// Redundant Load Elimination Pass
// Eliminates duplicate scalar tt.load operations in fused Triton kernels
// where the same pointer is loaded multiple times with no intervening store.
// This commonly arises after kernel fusion where each original kernel
// independently loaded shared invariants (e.g., positions, strides) in
// its prologue.

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREDUNDANTLOADELIMINATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// =========================================================================
// Redundant Load Elimination Pattern
//
// Matches:
//   %a = tt.load %ptr        (scalar load)
//   ... (no tt.store to %ptr)
//   %b = tt.load %ptr        <- redundant; replace uses of %b with %a
//
// Only scalar (non-tensor pointer) loads are handled.  Tensor loads may
// have per-element masks and are left to CSE / other passes.
// =========================================================================
struct RedundantScalarLoadPattern : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Value loadPtr = loadOp.getPtr();

    // Only handle scalar (0-D) pointers, i.e. !tt.ptr<T> not tensor<Nx!tt.ptr<T>>
    if (isa<RankedTensorType>(loadPtr.getType()))
      return failure();

    // Walk backwards from loadOp to find an earlier load of the same pointer.
    for (Operation *prevOp = loadOp->getPrevNode(); prevOp;
         prevOp = prevOp->getPrevNode()) {

      // A store to the same pointer invalidates forwarding – stop search.
      if (auto storeOp = dyn_cast<triton::StoreOp>(prevOp)) {
        if (storeOp.getPtr() == loadPtr)
          return failure();
      }

      // Found an earlier scalar load of the same pointer with no intervening
      // store – replace the current load with the earlier result.
      if (auto prevLoad = dyn_cast<triton::LoadOp>(prevOp)) {
        if (prevLoad.getPtr() == loadPtr &&
            !isa<RankedTensorType>(prevLoad.getPtr().getType())) {
          rewriter.replaceOp(loadOp, prevLoad.getResult());
          return success();
        }
      }
    }
    return failure();
  }
};

// =========================================================================
// Pass body
// =========================================================================
struct TritonRedundantLoadEliminationPass
    : public impl::TritonRedundantLoadEliminationBase<
          TritonRedundantLoadEliminationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RedundantScalarLoadPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::triton
