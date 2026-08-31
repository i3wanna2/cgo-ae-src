//===- PreScaleQuery.cpp --------------------------------------*- C++ -*-===//
// Move scale before the loop by pre-scaling Q once and replacing
// mulf(dot(Q,K), scale) inside the loop with dot(Q*scale, K).
// Assumptions:
//  - Q tile is loaded/available outside the scf.for body.
//  - scale is a scalar f32 splat to f32 tensor used to scale the f32 dot.
//  - We pre-scale Q in f16 using truncf(scale) to f16 and a splat to Q's type.
// This changes rounding slightly but matches your perf intent.
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONPRESCALEQUERY
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
namespace {

static arith::MulFOp tryGetDotScaleMul(arith::MulFOp mul) {
  // Return mul if one operand comes from dot (possibly via truncf/extf),
  // the other is an f32 tensor splat/broadcast of a scalar.
  auto isDotF32 = [](Value v) -> triton::DotOp {
    if (auto d = v.getDefiningOp<triton::DotOp>())
      return d;
    if (auto ext = v.getDefiningOp<arith::ExtFOp>()) {
      if (auto tr = ext.getIn().getDefiningOp<arith::TruncFOp>())
        if (auto d2 = tr.getIn().getDefiningOp<triton::DotOp>())
          return d2;
    }
    return nullptr;
  };

  auto lhsDot = isDotF32(mul.getLhs());
  auto rhsDot = isDotF32(mul.getRhs());
  if (!lhsDot && !rhsDot)
    return nullptr;
  return mul;
}

static std::optional<Value> getScalarFromSplat(Value v) {
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  return std::nullopt;
}

struct PreScaleQueryInLoop : public OpRewritePattern<arith::MulFOp> {
  using OpRewritePattern<arith::MulFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulFOp mul,
                                PatternRewriter &rewriter) const override {
    auto parentFor = mul->getParentOfType<scf::ForOp>();
    if (!parentFor)
      return failure();

    // Identify dot and scale sides
    triton::DotOp dot = nullptr;
    Value scaleTensor;
    bool dotIsLhs = false;
    if ((dot = mul.getLhs().getDefiningOp<triton::DotOp>())) {
      scaleTensor = mul.getRhs();
      dotIsLhs = true;
    } else if ((dot = mul.getRhs().getDefiningOp<triton::DotOp>())) {
      scaleTensor = mul.getLhs();
    } else {
      // try via extf(truncf(dot))
      if (auto ext = mul.getLhs().getDefiningOp<arith::ExtFOp>()) {
        if (auto tr = ext.getIn().getDefiningOp<arith::TruncFOp>())
          if ((dot = tr.getIn().getDefiningOp<triton::DotOp>())) {
            scaleTensor = mul.getRhs();
            dotIsLhs = true;
          }
      }
      if (!dot) {
        if (auto ext = mul.getRhs().getDefiningOp<arith::ExtFOp>()) {
          if (auto tr = ext.getIn().getDefiningOp<arith::TruncFOp>())
            if ((dot = tr.getIn().getDefiningOp<triton::DotOp>())) {
              scaleTensor = mul.getLhs();
            }
        }
      }
      if (!dot)
        return failure();
    }

    // Ensure scaleTensor is a splat from a scalar f32
    auto shaped = dyn_cast<ShapedType>(scaleTensor.getType());
    if (!shaped || !isa<FloatType>(shaped.getElementType()))
      return failure();
    auto scalarOpt = getScalarFromSplat(scaleTensor);
    if (!scalarOpt)
      return failure();
    Value scalarF32 = *scalarOpt;
    if (!isa<FloatType>(scalarF32.getType()))
      return failure();

    // Q must be defined outside loop
    Value q = dot.getA();
    auto qType = dyn_cast<ShapedType>(q.getType());
    if (!qType || !isa<FloatType>(qType.getElementType()))
      return failure();
    if (parentFor->isAncestor(q.getDefiningOp()))
      return failure();

    // Build scaled Q outside the loop (before for)
    OpBuilder b(parentFor);
    b.setInsertionPoint(parentFor);
    // Convert scalar to f16 and splat to Q's shaped type
    auto f16Ty = Float16Type::get(rewriter.getContext());
    Value scalarF16 = b.create<arith::TruncFOp>(parentFor.getLoc(), f16Ty, scalarF32);
    Value scaleF16Tensor = b.create<triton::SplatOp>(parentFor.getLoc(), qType, scalarF16);
    Value scaledQ = b.create<arith::MulFOp>(parentFor.getLoc(), qType, q, scaleF16Tensor);

    // Create new dot inside loop with scaled Q
    rewriter.setInsertionPoint(dot);
    auto newDot = rewriter.create<triton::DotOp>(dot.getLoc(), dot.getType(),
                                                 scaledQ, dot.getB(), dot.getC(),
                                                 dot.getInputPrecisionAttr());

    // Replace the mul result with new dot result
    rewriter.replaceOp(mul, newDot.getResult());
    return success();
  }
};

struct PreScaleQueryPass : public impl::TritonPreScaleQueryBase<PreScaleQueryPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PreScaleQueryInLoop>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton
