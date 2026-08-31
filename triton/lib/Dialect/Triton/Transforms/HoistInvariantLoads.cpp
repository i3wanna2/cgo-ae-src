//===----------------------------------------------------------------------===//
// HoistInvariantLoads Pass
//
// This pass hoists loop-invariant load operations out of scf.for loops.
// It specifically targets patterns where:
// 1. A tt.load operation inside a loop doesn't depend on loop iteration
// variable
// 2. The loaded value can be safely moved before the loop
// 3. The pointer calculation is loop-invariant
//
// This optimization is particularly useful for attention mechanisms where
// Q matrix can be loaded once before the K/V loop.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONHOISTINVARIANTLOADS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Check if a value is defined outside the given loop
static bool isDefinedOutsideLoop(Value value, scf::ForOp forOp) {
  if (!value) {
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return true; // Block argument, conservatively assume it's outside
  }

  bool isAncestor = forOp->isAncestor(defOp);
  return !isAncestor;
}

// Check if an operation is loop-invariant (all operands are defined outside)
static bool isLoopInvariant(Operation *op, scf::ForOp forOp) {
  // Check if the operation itself is outside the loop
  if (!forOp->isAncestor(op))
    return true;

  // Check all operands
  for (Value operand : op->getOperands()) {
    if (!isDefinedOutsideLoop(operand, forOp))
      return false;
  }

  return true;
}

// Recursively collect all operations needed to compute a value
static void collectDependentOps(Value value, scf::ForOp forOp,
                                llvm::DenseSet<Operation *> &opsToHoist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return;

  if (opsToHoist.contains(defOp))
    return;

  // Don't hoist if it depends on loop iteration variable or loop-variant values
  if (!isLoopInvariant(defOp, forOp))
    return;

  opsToHoist.insert(defOp);

  // Recursively collect dependencies
  for (Value operand : defOp->getOperands()) {
    collectDependentOps(operand, forOp, opsToHoist);
  }
}

// Check if a load operation is safe to hoist
static bool isSafeToHoist(triton::LoadOp loadOp, scf::ForOp forOp) {
  // The load must be inside the loop
  if (!forOp->isAncestor(loadOp))
    return false;

  // Check if pointer is loop-invariant
  Value ptr = loadOp.getPtr();
  if (!isDefinedOutsideLoop(ptr, forOp))
    return false;

  // Check if mask (if present) is loop-invariant
  if (Value mask = loadOp.getMask()) {
    if (!isDefinedOutsideLoop(mask, forOp))
      return false;
  }

  // Check if other (if present) is loop-invariant
  if (Value other = loadOp.getOther()) {
    if (!isDefinedOutsideLoop(other, forOp))
      return false;
  }

  // TODO: Add more safety checks:
  // - Check for aliasing with stores inside the loop
  // - Check for side effects

  return true;
}

// Walk up pointer-producing ops to find the root base pointer (usually a
// function argument of type !tt.ptr<...>). We ignore indexing ops such as
// addptr/broadcast/splat when tracing the root.
static Value getRootPtr(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (auto addp = dyn_cast<triton::AddPtrOp>(op)) {
      v = addp.getPtr();
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      v = splat.getSrc();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(op)) {
      v = bcast.getSrc();
      continue;
    }
    // Unknown op defining pointer value; stop here.
    break;
  }
  return v;
}

// Conservatively check if there is any store inside the loop that aliases the
// same base pointer as the given load. If yes, we must not hoist.
static bool aliasesStoreInLoop(triton::LoadOp loadOp, scf::ForOp forOp) {
  Value loadRoot = getRootPtr(loadOp.getPtr());
  bool foundAlias = false;
  forOp.walk([&](triton::StoreOp storeOp) {
    Value storeRoot = getRootPtr(storeOp.getPtr());
    if (storeRoot && loadRoot && storeRoot == loadRoot)
      foundAlias = true;
  });
  return foundAlias;
}

//===----------------------------------------------------------------------===//
// Main Pass Implementation
//===----------------------------------------------------------------------===//

struct HoistInvariantLoadsPass
    : public impl::TritonHoistInvariantLoadsBase<HoistInvariantLoadsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](scf::ForOp forOp) { hoistLoadsFromLoop(forOp); });
  }

private:
  void hoistLoadsFromLoop(scf::ForOp forOp) {
    // Collect all load operations inside the loop
    SmallVector<triton::LoadOp> loadsToHoist;

    int totalLoads = 0;
    forOp.walk([&](triton::LoadOp loadOp) {
      totalLoads++;

      Value ptr = loadOp.getPtr();
      bool ptrOutside = isDefinedOutsideLoop(ptr, forOp);

      if (Value mask = loadOp.getMask()) {

        bool maskOutside = isDefinedOutsideLoop(mask, forOp);
      }

      bool safe =
          isSafeToHoist(loadOp, forOp) && !aliasesStoreInLoop(loadOp, forOp);

      if (safe) {
        loadsToHoist.push_back(loadOp);
      }
    });

    if (loadsToHoist.empty())
      return;

    OpBuilder builder(forOp);

    for (triton::LoadOp loadOp : loadsToHoist) {
      // Since the pointer and mask are already loop-invariant (defined
      // outside), we can directly create a new load before the loop
      builder.setInsertionPoint(forOp);

      // Simply clone the load operation before the loop
      // clone() will handle all attributes and operands correctly
      auto *clonedOp = builder.clone(*loadOp.getOperation());
      auto newLoad = cast<triton::LoadOp>(clonedOp);

      // Replace all uses of the original load with the hoisted one
      loadOp.getResult().replaceAllUsesWith(newLoad.getResult());

      // Remove the original load
      loadOp->erase();
    }
  }
};

} // namespace
} // namespace mlir::triton
