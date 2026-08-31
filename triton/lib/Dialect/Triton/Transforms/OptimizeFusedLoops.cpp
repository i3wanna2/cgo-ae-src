#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONOPTIMIZEFUSEDLOOPS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Pass的主体
struct TritonOptimizeFusedLoopsPass
    : public impl::TritonOptimizeFusedLoopsBase<TritonOptimizeFusedLoopsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Iterate over every function in the module
    for (triton::FuncOp funcOp : module.getOps<triton::FuncOp>()) {

      // Always get a fresh list of loops in each iteration
      SmallVector<scf::ForOp> loops;
      // Note: We are now operating on the body of funcOp, not the module.
      if (funcOp.getBody().empty())
        continue;

      for (Operation &op : funcOp.getBody().front()) {
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          loops.push_back(loop);
        }
      }

      if (loops.size() < 2) {
        break; // Fewer than 2 loops, nothing to fuse in this function for
               // this iteration.
      }

      // Iterate through adjacent pairs of loops to find fusion candidates
      for (size_t i = 0; i < loops.size() - 1; ++i) {
        scf::ForOp loop1 = loops[i];
        scf::ForOp loop2 = loops[i + 1];

        // Legality Check: Iteration spaces must be identical
        if (loop1.getLowerBound() != loop2.getLowerBound() ||
            loop1.getUpperBound() != loop2.getUpperBound() ||
            loop1.getStep() != loop2.getStep()) {
        continue; 
        }
        bool hasDependency = false;
        for (Value res : loop1.getResults()) {
          for (Operation *user : res.getUsers()) {
            // 如果 loop1 结果的一个用户，其祖先是 loop2，说明存在依赖
            if (loop2->isAncestor(user)) {
              hasDependency = true;
              break;
            }
          }
          if (hasDependency)
            break;
        }

        if (hasDependency) {
          // 存在依赖，不能融合，跳过这对循环
        //   printf("loop [%d] not fusable due to dependency\n", (int)i);
          continue;
        }
        // printf("loop [%d] fusable\n", (int)i);
        if (loop1->getNextNode() != loop2.getOperation()) {
            // 它们不相邻，检查并移动中间的所有操作
            SmallVector<Operation*> opsToHoist;
            bool hoistSafe = true;
            for (Operation *op = loop1->getNextNode(); op && op != loop2.getOperation(); op = op->getNextNode()) {
                for (Value operand : op->getOperands()) {
                    if (operand.getDefiningOp() == loop1) {
                        hoistSafe = false;
                        break;
                    }
                }
                if (!hoistSafe) break;
                opsToHoist.push_back(op);
            }
            if (!hoistSafe) {
                continue; 
            }
            // 将所有中间操作移动到 loop1 之前
            for (Operation *op : opsToHoist) {
                op->moveBefore(loop1);
            }
        }

        // Fusion is legal, proceed
        OpBuilder builder(loop1);

        SmallVector<Value> newInitArgs;
        newInitArgs.append(loop1.getInitArgs().begin(),
                           loop1.getInitArgs().end());
        newInitArgs.append(loop2.getInitArgs().begin(),
                           loop2.getInitArgs().end());

        auto fusedLoop = builder.create<scf::ForOp>(
            loop1.getLoc(), loop1.getLowerBound(), loop1.getUpperBound(),
            loop1.getStep(), newInitArgs);

        Block *newBody = fusedLoop.getBody();
        newBody->getOperations().clear();
        builder.setInsertionPointToStart(newBody);

        IRMapping mapper;

        mapper.map(loop1.getInductionVar(), fusedLoop.getInductionVar());
        mapper.map(loop2.getInductionVar(), fusedLoop.getInductionVar());

        unsigned loop1IterArgsCount = loop1.getNumRegionIterArgs();
        for (unsigned j = 0; j < loop1IterArgsCount; ++j) {
          mapper.map(loop1.getRegionIterArgs()[j],
                     fusedLoop.getRegionIterArgs()[j]);
        }
        for (unsigned j = 0; j < loop2.getNumRegionIterArgs(); ++j) {
          mapper.map(loop2.getRegionIterArgs()[j],
                     fusedLoop.getRegionIterArgs()[loop1IterArgsCount + j]);
        }

        for (Operation &op : *loop1.getBody()) {
          if (!isa<scf::YieldOp>(op))
            builder.clone(op, mapper);
        }
        for (Operation &op : *loop2.getBody()) {
          if (!isa<scf::YieldOp>(op))
            builder.clone(op, mapper);
        }

        auto yield1 = cast<scf::YieldOp>(loop1.getBody()->getTerminator());
        auto yield2 = cast<scf::YieldOp>(loop2.getBody()->getTerminator());
        SmallVector<Value> finalYieldOperands;
        for (Value operand : yield1.getOperands()) {
          finalYieldOperands.push_back(mapper.lookup(operand));
        }
        for (Value operand : yield2.getOperands()) {
          finalYieldOperands.push_back(mapper.lookup(operand));
        }
        builder.create<scf::YieldOp>(fusedLoop.getLoc(), finalYieldOperands);

        unsigned loop1ResultsCount = loop1.getNumResults();
        loop1.replaceAllUsesWith(
            fusedLoop.getResults().take_front(loop1ResultsCount));
        loop2.replaceAllUsesWith(
            fusedLoop.getResults().drop_front(loop1ResultsCount));

        loop1.erase();
        loop2.erase();

        break;
      }
    } // end while(changed)
  }   // end for(funcOp...)
};
} // namespace
} // namespace mlir::triton
