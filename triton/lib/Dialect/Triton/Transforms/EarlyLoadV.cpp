// Move V loads earlier in the softmax loop to overlap latency.
//
// This pass identifies tt.load operations that feed the B (RHS) operand of
// tt.dot within scf.for loops, and safely clones their pointer-computation DAG
// to insert an early load right after the first dot (or first load) in the
// loop body. This avoids dominance violations that moving could cause.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::triton;

#define DEBUG_TYPE "triton-early-load-v"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONEARLYLOADV
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Whitelist of pure, cheap ops we can clone for pointer computation
static bool isCloneableAddrOp(Operation *op) {
  if (!op)
    return false;
  Dialect *d = op->getDialect();
  if (!d)
    return false;
  if (isa<arith::ArithDialect, math::MathDialect>(d))
    return true;
  if (isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
          triton::AddPtrOp, triton::MakeRangeOp>(op))
    return true;
  return false;
}

// Collect a cloneable post-order for the DAG that computes value 'v'.
// Returns false if a non-cloneable dependency is defined after 'anchor'.
static bool collectCloneableDAG(Value v, Block *loopBody, Operation *anchor,
                                SmallVectorImpl<Operation *> &postOrder,
                                SmallPtrSetImpl<Operation *> &seen) {
  if (!v)
    return true;
  if (isa<BlockArgument>(v))
    return true; // loop-carried args are available

  Operation *def = v.getDefiningOp();
  if (!def)
    return true; // external to this function
  if (def->getBlock() != loopBody)
    return true; // defined outside loop, dominates
  if (seen.contains(def))
    return true;

  if (isCloneableAddrOp(def)) {
    for (Value opnd : def->getOperands()) {
      if (!collectCloneableDAG(opnd, loopBody, anchor, postOrder, seen))
        return false;
    }
    seen.insert(def);
    postOrder.push_back(def);
    return true;
  }

  // If not cloneable, require it to dominate the anchor
  return def->isBeforeInBlock(anchor);
}

struct TritonEarlyLoadVPass
    : public impl::TritonEarlyLoadVBase<TritonEarlyLoadVPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();

    root->walk([&](scf::ForOp forOp) {
      Block *body = forOp.getBody();
      if (!body || body->empty())
        return;

      // Find the first dot in the loop
      triton::DotOp firstDot = nullptr;
      for (Operation &it : *body) {
        if (auto dot = dyn_cast<triton::DotOp>(&it)) {
          firstDot = dot;
          break;
        }
      }
      if (!firstDot)
        return;

      // Find V loads that are used by dots with truncf in A (attention pattern)
      SmallVector<std::pair<triton::LoadOp, triton::DotOp>, 4> vLoadDotPairs;
      for (Operation &it : *body) {
        if (auto dot = dyn_cast<triton::DotOp>(&it)) {
          // Check if A comes from truncf (indicates attention computation)
          if (dot.getA().getDefiningOp<arith::TruncFOp>()) {
            // Check if B is directly a load
            if (auto vLoad = dot.getB().getDefiningOp<triton::LoadOp>()) {
              if (vLoad->getBlock() == body && !vLoad.getMask() && !vLoad.getOther()) {
                vLoadDotPairs.push_back({vLoad, dot});
              }
            }
          }
        }
      }
      if (vLoadDotPairs.empty())
        return;

      OpBuilder builder(ctx);
      
      for (auto [vLoad, dotOp] : vLoadDotPairs) {
        // Collect cloneable DAG for the pointer
        Value ptr = vLoad.getPtr();
        SmallVector<Operation *, 16> topo;
        SmallPtrSet<Operation *, 32> seen;
        // Use firstDot as anchor to ensure all cloned ops dominate it
        if (!collectCloneableDAG(ptr, body, firstDot, topo, seen))
          continue;

        // Sanity check: any non-cloned in-loop operand must dominate firstDot
        bool ok = true;
        for (Operation *orig : topo) {
          for (Value opnd : orig->getOperands()) {
            if (Operation *d = opnd.getDefiningOp()) {
              if (d->getBlock() == body && !seen.contains(d) && !d->isBeforeInBlock(firstDot)) {
                ok = false; 
                break;
              }
            }
          }
          if (!ok) break;
        }
        if (!ok)
          continue;

        // Clone DAG right BEFORE firstDot and construct new load
        IRMapping map;
        builder.setInsertionPoint(firstDot);
        Operation *lastCloned = nullptr;
        for (Operation *orig : topo) {
          Operation *cl = builder.clone(*orig, map);
          lastCloned = cl;
          builder.setInsertionPointAfter(cl);
        }

        Value newPtr = map.lookupOrNull(ptr);
        if (!newPtr)
          continue;

        IRMapping loadMap = map;
        loadMap.map(vLoad.getPtr(), newPtr);
        // Place the new load right before the first dot
        builder.setInsertionPoint(firstDot);
        Operation *newLoadOp = builder.clone(*vLoad.getOperation(), loadMap);
        Value newV = newLoadOp->getResult(0);

        // Replace all uses of the original V load with the new early-loaded value
        vLoad.getResult().replaceAllUsesWith(newV);
        
        // Erase the original load operation
        vLoad.erase();
      }
    });
  }
};

} // namespace
} // namespace mlir::triton