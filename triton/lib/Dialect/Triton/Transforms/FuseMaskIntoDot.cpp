// Fuse additive -inf/0 mask into tt.dot's C operand.
// Pattern examples:
//   - addf(truncf(tt.dot(a,b,0)), mask_f16) (optionally extf afterwards)
//   - addf(tt.dot(a,b,0), mask_f32)
// Be conservative and only fuse when C is a zero tensor constant and mask is
// built as select(cond, -inf, 0) or select(cond, 0, -inf) in any fp type.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-fuse-mask-into-dot"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_TRITONFUSEMASKINTODOT
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Get element type width in bits
static unsigned getElementTypeWidth(Type type) {
  if (auto shapedType = llvm::dyn_cast<ShapedType>(type))
    type = shapedType.getElementType();
  if (auto floatType = llvm::dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return 0;
}

static bool isZeroLikeTensor(Value v) {
  // Accept dense<0> constants or splats of 0 in any float type
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast_or_null<DenseElementsAttr>(cst.getValue())) {
      if (attr.isSplat()) {
        if (auto fAttr = llvm::dyn_cast<FloatAttr>(attr.getSplatValue<Attribute>()))
          return fAttr.getValue().isZero();
      }
    }
  }
  return false;
}

static bool isNegInfZeroSelect(arith::SelectOp sel, Value &negInfOrZeroFirst,
                               bool &firstIsNegInf) {
  // Recognize select(cond, X, Y) where X/Y are -inf and 0 (any order)
  auto lhs = sel.getTrueValue();
  auto rhs = sel.getFalseValue();
  auto isZeroTensor = [](Value v) -> bool {
    // Case 1: dense splat 0 tensor
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = llvm::dyn_cast_or_null<DenseElementsAttr>(cst.getValue()))
        if (attr.isSplat()) {
          if (auto f = llvm::dyn_cast<FloatAttr>(attr.getSplatValue<Attribute>()))
            return f.getValue().isZero();
        }
    }
    // Case 2: splat of scalar 0.0
    if (auto splat = v.getDefiningOp<triton::SplatOp>()) {
      if (auto sc = splat.getSrc().getDefiningOp<arith::ConstantFloatOp>()) {
        if (auto fattr = dyn_cast<FloatAttr>(sc.getValueAttr()))
          return fattr.getValue().isZero();
      }
      if (auto sc2 = splat.getSrc().getDefiningOp<arith::ConstantOp>())
        if (auto f = dyn_cast<FloatAttr>(sc2.getValue()))
          return f.getValue().isZero();
    }
    return false;
  };
  auto isNegInfTensor = [](Value v) -> bool {
    // Case 1: dense splat -inf
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = llvm::dyn_cast_or_null<DenseElementsAttr>(cst.getValue()))
        if (attr.isSplat()) {
          if (auto f = llvm::dyn_cast<FloatAttr>(attr.getSplatValue<Attribute>()))
            return f.getValue().isInfinity() && f.getValue().isNegative();
        }
    }
    // Case 2: splat of scalar -inf
    if (auto splat = v.getDefiningOp<triton::SplatOp>()) {
      if (auto sc = splat.getSrc().getDefiningOp<arith::ConstantFloatOp>()) {
        if (auto fattr = dyn_cast<FloatAttr>(sc.getValueAttr())) {
          auto apf = fattr.getValue();
          return apf.isInfinity() && apf.isNegative();
        }
      }
      if (auto sc2 = splat.getSrc().getDefiningOp<arith::ConstantOp>())
        if (auto f = dyn_cast<FloatAttr>(sc2.getValue())) {
          auto apf = f.getValue();
          return apf.isInfinity() && apf.isNegative();
        }
    }
    return false;
  };

  if (isNegInfTensor(lhs) && isZeroTensor(rhs)) {
    negInfOrZeroFirst = lhs;
    firstIsNegInf = true;
    return true;
  }
  if (isZeroTensor(lhs) && isNegInfTensor(rhs)) {
    negInfOrZeroFirst = lhs;
    firstIsNegInf = false;
    return true;
  }
  return false;
}

// addf(X, mask) -> dot(a,b,mask) if X is dot(a,b,zeroC) possibly with truncf/extf
struct FuseAddMaskIntoDot : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

  // 递归遍历定义链，查找dot操作
  triton::DotOp findDotOp(Value v) const {
    if (auto dot = v.getDefiningOp<triton::DotOp>())
      return dot;
    if (auto trunc = v.getDefiningOp<arith::TruncFOp>())
      return findDotOp(trunc.getIn());
    if (auto ext = v.getDefiningOp<arith::ExtFOp>())
      return findDotOp(ext.getIn());
    if (auto add = v.getDefiningOp<arith::AddFOp>())
      return findDotOp(add.getLhs());
    return nullptr;
  }

  // 检查是否是 mask 模式（包括更多变体）
  bool isMaskPattern(Value v, Value &maskValue) const {
    // 直接的 select 模式
    if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
      Value tmp;
      bool firstIsNegInf;
      if (isNegInfZeroSelect(sel, tmp, firstIsNegInf)) {
        maskValue = sel;
        return true;
      }
    }

    // 处理 softmax 中的 mask
    if (auto extf = v.getDefiningOp<arith::ExtFOp>()) {
      return isMaskPattern(extf.getIn(), maskValue);
    }
    if (auto truncf = v.getDefiningOp<arith::TruncFOp>()) {
      return isMaskPattern(truncf.getIn(), maskValue);
    }

    return false;
  }

  LogicalResult matchAndRewrite(arith::AddFOp add,
                                PatternRewriter &rewriter) const override {
    Value lhs = add.getLhs();
    Value rhs = add.getRhs();

    // 查找 dot 和 mask
    Value maskValue;
    Value dotSide;
    bool foundPattern = false;

    // 检查左右操作数
    if (isMaskPattern(lhs, maskValue)) {
      dotSide = rhs;
      foundPattern = true;
    } else if (isMaskPattern(rhs, maskValue)) {
      dotSide = lhs;
      foundPattern = true;
    }

    if (!foundPattern)
      return failure();

    // 在操作链中查找 dot
    auto dot = findDotOp(dotSide);
    if (!dot)
      return failure();

    // C of dot must be zero-like
    if (!isZeroLikeTensor(dot.getC()))
      return failure();

    // 确保类型/形状兼容性：mask 必须匹配 dot 累加器形状
    RankedTensorType accTy = llvm::dyn_cast<RankedTensorType>(dot.getD().getType());
    RankedTensorType maskTy = llvm::dyn_cast<RankedTensorType>(maskValue.getType());
    if (!accTy || !maskTy)
      return failure();
    if (maskTy.getShape() != accTy.getShape())
      return failure();

    // 将 mask 转换为累加器的元素类型（如果需要）
    Value maskVal = maskValue;
    if (maskTy.getElementType() != accTy.getElementType()) {
      maskVal = rewriter.create<arith::ExtFOp>(add.getLoc(),
                  RankedTensorType::get(maskTy.getShape(), accTy.getElementType()),
                  maskVal);
    }

    // 构建新的 dot 操作，保持类型和精度
    auto newDot = rewriter.create<triton::DotOp>(
        dot.getLoc(), accTy,
        dot.getA(), dot.getB(), maskVal,
        dot.getInputPrecisionAttr(),
        dot.getMaxNumImpreciseAccAttr());

    // 处理可能的类型转换
    Value resultValue = newDot.getResult();

    // 如果需要，插入必要的类型转换
    if (add.getType() != accTy) {
      // 如果结果类型是窄类型（如f16），需要截断
      if (getElementTypeWidth(add.getType()) < getElementTypeWidth(accTy)) {
        resultValue = rewriter.create<arith::TruncFOp>(
            add.getLoc(), add.getType(), resultValue);
      } else {
        // 如果结果类型是宽类型，需要扩展
        resultValue = rewriter.create<arith::ExtFOp>(
            add.getLoc(), add.getType(), resultValue);
      }
    }

    // 替换原始操作
    rewriter.replaceOp(add, resultValue);

    return failure();
  }
};

struct TritonFuseMaskIntoDotPass
    : public impl::TritonFuseMaskIntoDotBase<TritonFuseMaskIntoDotPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseAddMaskIntoDot>(ctx);
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace triton
} // namespace mlir
