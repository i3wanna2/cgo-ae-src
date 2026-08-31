#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-fuse-softmax-gemm"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONFUSESOFTMAXGEMM
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Helper to check if a value is used within a specific loop
bool isUsedInLoop(Value val, scf::ForOp loop) {
  for (auto *user : val.getUsers()) {
    if (loop->isAncestor(user))
      return true;
  }
  return false;
}

// Helper to find operation of specific type in a block
template <typename OpTy>
OpTy findOpInBlock(Block *block) {
  OpTy result;
  block->walk([&](OpTy op) {
    if (!result)
      result = op;
  });
  return result;
}

// Check if the pattern matches: load -> extf -> sub -> exp -> divf -> select -> truncf -> dot
struct SoftmaxGemmPattern {
  triton::LoadOp loadOp;
  arith::ExtFOp extfOp;
  arith::SubFOp subOp;
  math::ExpOp expOp;
  arith::DivFOp divOp;
  arith::SelectOp selectOp;
  arith::TruncFOp truncOp;
  triton::DotOp dotOp;
  
  bool isValid() const {
    return loadOp && extfOp && subOp && expOp && divOp && 
           selectOp && truncOp && dotOp;
  }
};

// Analyze the second loop to find the softmax + gemm pattern
SoftmaxGemmPattern analyzeSoftmaxGemmLoop(scf::ForOp loop) {
  SoftmaxGemmPattern pattern;
  
  Block *body = loop.getBody();
  
  // Find operations in reverse order (from dot backwards)
  pattern.dotOp = findOpInBlock<triton::DotOp>(body);
  if (!pattern.dotOp)
    return pattern;
  
  // The first operand of dot should come from truncf
  auto lhs = pattern.dotOp.getA();
  pattern.truncOp = lhs.getDefiningOp<arith::TruncFOp>();
  if (!pattern.truncOp)
    return pattern;
  
  // truncf comes from select
  pattern.selectOp = pattern.truncOp.getOperand().getDefiningOp<arith::SelectOp>();
  if (!pattern.selectOp)
    return pattern;
  
  // select's true value comes from divf
  pattern.divOp = pattern.selectOp.getTrueValue().getDefiningOp<arith::DivFOp>();
  if (!pattern.divOp)
    return pattern;
  
  // divf's lhs comes from exp
  pattern.expOp = pattern.divOp.getLhs().getDefiningOp<math::ExpOp>();
  if (!pattern.expOp)
    return pattern;
  
  // exp's operand comes from subf
  pattern.subOp = pattern.expOp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!pattern.subOp)
    return pattern;
  
  // subf's lhs comes from extf
  pattern.extfOp = pattern.subOp.getLhs().getDefiningOp<arith::ExtFOp>();
  if (!pattern.extfOp)
    return pattern;
  
  // extf's operand comes from load
  pattern.loadOp = pattern.extfOp.getOperand().getDefiningOp<triton::LoadOp>();
  if (!pattern.loadOp)
    return pattern;
  
  return pattern;
}

struct FuseSoftmaxGemmPass
    : public impl::TritonFuseSoftmaxGemmBase<FuseSoftmaxGemmPass> {
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Walk through all functions
    module.walk([&](triton::FuncOp func) {
      SmallVector<std::pair<scf::ForOp, scf::ForOp>> loopPairs;
      
      // Find consecutive for loops
      func.walk([&](scf::ForOp firstLoop) {
        // Check if first loop has 2 results (max and sum)
        if (firstLoop.getNumResults() != 2)
          return;
        
        // Find the next scf.for in the same block
        Operation *nextOp = firstLoop->getNextNode();
        while (nextOp) {
          if (auto secondLoop = dyn_cast<scf::ForOp>(nextOp)) {
            // Check if second loop uses results from first loop
            auto maxVal = firstLoop.getResult(0);
            auto sumVal = firstLoop.getResult(1);
            
            if (isUsedInLoop(maxVal, secondLoop) && 
                isUsedInLoop(sumVal, secondLoop)) {
              loopPairs.push_back({firstLoop, secondLoop});
            }
            break;
          }
          nextOp = nextOp->getNextNode();
        }
      });
      
      // Process each pair
      for (auto [firstLoop, secondLoop] : loopPairs) {
        if (failed(fuseSoftmaxGemm(firstLoop, secondLoop))) {
          LLVM_DEBUG(llvm::dbgs() << "Failed to fuse softmax-gemm pattern\n");
        }
      }
    });
  }
  
private:
  LogicalResult fuseSoftmaxGemm(scf::ForOp firstLoop, scf::ForOp secondLoop) {
    // Analyze the second loop
    SoftmaxGemmPattern pattern = analyzeSoftmaxGemmLoop(secondLoop);
    if (!pattern.isValid()) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern not valid\n");
      return failure();
    }
    
    auto maxVal = firstLoop.getResult(0);  // %16#0
    auto sumVal = firstLoop.getResult(1);  // %16#1
    
    OpBuilder builder(secondLoop);
    
    // Step 1: Remove the division before dot
    // The chain is: load -> extf -> sub -> exp -> div -> select -> trunc -> dot
    // We want: load -> extf -> sub -> exp -> select -> trunc -> dot
    //    then add: dot_result / broadcast(sum) outside the loop
    
    // Get the mask and false value from select
    Value mask = pattern.selectOp.getCondition();
    Value falseValue = pattern.selectOp.getFalseValue();
    
    // Create new select that uses exp result directly (without div)
    builder.setInsertionPoint(pattern.selectOp);
    auto newSelect = builder.create<arith::SelectOp>(
        pattern.selectOp.getLoc(),
        mask,
        pattern.expOp.getResult(),  // Use exp result directly
        falseValue
    );
    
    // Update truncf to use new select
    pattern.truncOp.getOperation()->setOperand(0, newSelect.getResult());
    
    // Remove old div and select
    pattern.selectOp.replaceAllUsesWith(newSelect.getResult());
    pattern.selectOp.erase();
    pattern.divOp.erase();
    
    // Step 2: Add division after the loop
    builder.setInsertionPointAfter(secondLoop);
    
    // Get the result of the second loop (dot result)
    Value loopResult = secondLoop.getResult(0);
    auto resultType = cast<RankedTensorType>(loopResult.getType());
    
    // Broadcast sum to match result shape
    // sum is tensor<128x1xf32>, we need to broadcast to tensor<128x128xf32>
    auto sumType = cast<RankedTensorType>(sumVal.getType());
    
    // Create broadcast for sum
    auto broadcastSum = builder.create<triton::BroadcastOp>(
        secondLoop.getLoc(),
        resultType,
        sumVal
    );
    
    // Create division
    auto finalDiv = builder.create<arith::DivFOp>(
        secondLoop.getLoc(),
        loopResult,
        broadcastSum.getResult()
    );
    
    // Replace all uses of loop result with the divided result
    loopResult.replaceAllUsesExcept(finalDiv.getResult(), finalDiv);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused softmax-gemm pattern\n");
    return success();
  }
};

} // namespace

} // namespace mlir::triton
