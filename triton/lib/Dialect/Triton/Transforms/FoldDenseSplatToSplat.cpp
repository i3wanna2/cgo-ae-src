#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONFOLDDENSESPLATTOSPLAT
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
namespace {

struct FoldDenseSplatToSplatPass
    : public impl::TritonFoldDenseSplatToSplatBase<FoldDenseSplatToSplatPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<arith::ConstantOp, 8> worklist;
    m.walk([&](arith::ConstantOp cst) {
      auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!attr || !attr.isSplat())
        return;
      auto shapedTy = dyn_cast<ShapedType>(cst.getType());
      if (!shapedTy || !shapedTy.hasStaticShape())
        return;
      // Only non-scalar tensors are interesting
      if (shapedTy.getRank() == 0)
        return;
      worklist.push_back(cst);
    });

    for (auto cst : worklist) {
      auto attr = cast<DenseElementsAttr>(cst.getValue());
      Attribute elemVal = attr.getSplatValue<Attribute>();
      auto typedElemVal = dyn_cast<TypedAttr>(elemVal);
      if (!typedElemVal)
        continue;
      OpBuilder b(&getContext());
      b.setInsertionPoint(cst);
      Value scalar = b.create<arith::ConstantOp>(cst.getLoc(), typedElemVal);
      Value splat = b.create<triton::SplatOp>(cst.getLoc(), cst.getType(), scalar);
      cst.getResult().replaceAllUsesWith(splat);
      cst.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton
