//===- EliminateCastRoundtrip.cpp -------------------------------*- C++ -*-===//
//
// Triton: Eliminate float cast round-trips like truncf(extf(x)) and
// optionally extf(truncf(x)) when desired for performance.
// This pass focuses on:
//  - truncf(extf(x)) -> x        (always safe)
//  - extf(truncf(x)) -> x        (semantics-changing; enabled here per request)
//  - collapse chains like extf(extf(x)) and truncf(truncf(x))
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONELIMINATECASTROUNDTRIP
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
namespace {

static Type getElemType(Type t) {
  if (auto st = dyn_cast<ShapedType>(t))
    return st.getElementType();
  return t;
}

// Fold truncf(extf(x narrow -> wide) wide -> narrow) -> x
struct TruncOfExtFold : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp trunc,
                                PatternRewriter &rewriter) const override {
    auto ext = trunc.getIn().getDefiningOp<arith::ExtFOp>();
    if (!ext)
      return failure();

    auto srcType = dyn_cast<FloatType>(getElemType(ext.getIn().getType()));
    auto midType = dyn_cast<FloatType>(getElemType(ext.getResult().getType()));
    auto dstType = dyn_cast<FloatType>(getElemType(trunc.getResult().getType()));
    if (!srcType || !midType || !dstType)
      return failure();

    // ext: src -> mid, trunc: mid -> dst. We only fold when src == dst.
    if (srcType != dstType)
      return failure();

    rewriter.replaceOp(trunc, ext.getIn());
    return success();
  }
};

// Eliminate extf(truncf(x wide -> narrow) narrow -> wide) -> x
// NOTE: This changes numerical semantics by removing the rounding to "narrow".
// It is being enabled intentionally to assess performance impact.
struct ExtOfTruncElim : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp ext,
                                PatternRewriter &rewriter) const override {
    auto trunc = ext.getIn().getDefiningOp<arith::TruncFOp>();
    if (!trunc)
      return failure();

    auto wideType = dyn_cast<FloatType>(getElemType(ext.getResult().getType()));
    auto midType = dyn_cast<FloatType>(getElemType(trunc.getResult().getType()));
    auto srcType = dyn_cast<FloatType>(getElemType(trunc.getIn().getType()));
    if (!wideType || !midType || !srcType)
      return failure();

    // trunc: wide(src) -> narrow(mid), ext: narrow(mid) -> wide(result)
    // Only when the final wide type matches the original src wide type
    // can we replace with the original value.
    if (wideType != srcType)
      return failure();

    rewriter.replaceOp(ext, trunc.getIn());
    return success();
  }
};

// Collapse extf(extf(x)) => extf(x) directly from source to final type.
struct CollapseExtChain : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp ext,
                                PatternRewriter &rewriter) const override {
    auto inner = ext.getIn().getDefiningOp<arith::ExtFOp>();
    if (!inner)
      return failure();

    auto srcType = dyn_cast<FloatType>(getElemType(inner.getIn().getType()));
    auto dstType = dyn_cast<FloatType>(getElemType(ext.getResult().getType()));
    if (!srcType || !dstType)
      return failure();

    rewriter.replaceOpWithNewOp<arith::ExtFOp>(ext, dstType, inner.getIn());
    return success();
  }
};

// Collapse truncf(truncf(x)) => truncf(x) directly to final type.
struct CollapseTruncChain : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp trunc,
                                PatternRewriter &rewriter) const override {
    auto inner = trunc.getIn().getDefiningOp<arith::TruncFOp>();
    if (!inner)
      return failure();

    auto srcType = dyn_cast<FloatType>(getElemType(inner.getIn().getType()));
    auto dstType = dyn_cast<FloatType>(getElemType(trunc.getResult().getType()));
    if (!srcType || !dstType)
      return failure();

    rewriter.replaceOpWithNewOp<arith::TruncFOp>(trunc, dstType, inner.getIn());
    return success();
  }
};

struct EliminateCastRoundtrip
    : public impl::TritonEliminateCastRoundtripBase<EliminateCastRoundtrip> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TruncOfExtFold, ExtOfTruncElim, CollapseExtChain, CollapseTruncChain>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

// Factory is generated in Passes.h.inc within the impl namespace.

} // namespace mlir::triton

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//
