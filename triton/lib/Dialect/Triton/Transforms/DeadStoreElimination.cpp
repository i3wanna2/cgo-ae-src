#include "mlir/Dialect/Arith/IR/Arith.h"
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

#define GEN_PASS_DEF_TRITONDEADSTOREELIMINATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// =========================================================================
// Dead Store Elimination Pattern
// 删除从未被 load 的 store 操作
// 
// 安全检查：
// - 如果 store 的基础指针是函数参数，检查是否有 temp_buffer 标记
// - 只有标记为临时缓冲区的参数才能删除 dead store
// - 非参数的 store 如果后续没有 load 也可以删除
// - 可以指定特定的 argument 序号，这些参数的 store 将被保护（不删除）
// =========================================================================
struct DeadStoreEliminationPattern : public OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  // 保护的参数序号集合
  const llvm::SmallDenseSet<int32_t, 16> &preserveArgs;

  DeadStoreEliminationPattern(MLIRContext *ctx, 
                              const llvm::SmallDenseSet<int32_t, 16> &args)
      : OpRewritePattern(ctx), preserveArgs(args) {}

  // 提取指针计算链的基础值
  // 需要追溯到最原始的函数参数或局部分配的指针
  // 
  // 支持两种模式：
  // 模式1: %arg -> tt.splat -> tt.addptr (直接splat函数参数)
  // 模式2: %arg -> tt.addptr -> tt.splat -> tt.addptr (先偏移再splat)
  Value getBasePointer(Value ptr) const {
    Value current = ptr;
    while (current) {
      if (auto addptrOp = current.getDefiningOp<triton::AddPtrOp>()) {
        // 继续追溯addptr的基础指针
        current = addptrOp.getPtr();
      } else if (auto splatOp = current.getDefiningOp<triton::SplatOp>()) {
        // 如果遇到splat，获取其源
        Value src = splatOp.getSrc();
        
        // 判断是否需要继续追溯：
        // 如果源是函数参数或没有定义操作（常量等），说明已经到达终点
        if (!src.getDefiningOp() || isa<BlockArgument>(src)) {
          return src;  // 模式1：tt.splat %arg -> 返回 %arg
        }
        
        // 否则继续追溯（可能是另一个addptr等）
        current = src;  // 模式2：tt.splat (%addptr %arg) -> 继续追溯到 %arg
      } else if (auto broadcastOp = current.getDefiningOp<triton::BroadcastOp>()) {
        // 如果遇到broadcast，继续追溯其源
        current = broadcastOp.getSrc();
      } else if (auto bitcastOp = current.getDefiningOp<triton::BitcastOp>()) {
        // 如果遇到bitcast，继续追溯其源
        current = bitcastOp.getSrc();
      } else {
        // 到达终点：函数参数、常量、或其他无法追溯的值
        return current;
      }
    }
    return current;
  }

  // 检查是否是函数参数
  bool isFunctionArgument(Value val) const {
    return isa<BlockArgument>(val) && 
           val.getParentBlock()->isEntryBlock();
  }

  // 检查参数是否在保护列表中
  bool isPreservedArgument(Value val) const {
    if (!isFunctionArgument(val))
      return false;
    
    auto blockArg = cast<BlockArgument>(val);
    unsigned argNum = blockArg.getArgNumber();
    return preserveArgs.count(static_cast<int32_t>(argNum)) > 0;
  }

  // 检查函数参数是否被标记为临时缓冲区
  bool isTempBufferArgument(Value val) const {
    if (!isFunctionArgument(val))
      return false;
    
    auto blockArg = cast<BlockArgument>(val);
    unsigned argNum = blockArg.getArgNumber();
    Operation *funcOp = blockArg.getOwner()->getParentOp();
    
    // 检查是否有 "argN.temp_buffer" 属性
    std::string attrName = "arg" + std::to_string(argNum) + ".temp_buffer";
    return funcOp->hasAttr(attrName);
  }

  // 检查基础指针在整个函数中是否被 load 过
  // 需要追踪所有从 basePtr 派生的指针（通过 splat/addptr/broadcast）
  bool basePointerEverLoaded(Value basePtr) const {
    SmallVector<Value> worklist;
    DenseSet<Value> visited;
    worklist.push_back(basePtr);
    visited.insert(basePtr);
    
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      
      for (Operation *user : current.getUsers()) {
        // 如果找到 load 操作，说明基础指针被使用了
        if (isa<triton::LoadOp>(user)) {
          return true;
        }
        
        // 追踪通过指针计算派生的值
        if (auto splatOp = dyn_cast<triton::SplatOp>(user)) {
          Value result = splatOp.getResult();
          if (!visited.count(result)) {
            visited.insert(result);
            worklist.push_back(result);
          }
        } else if (auto addptrOp = dyn_cast<triton::AddPtrOp>(user)) {
          Value result = addptrOp.getResult();
          if (!visited.count(result)) {
            visited.insert(result);
            worklist.push_back(result);
          }
        } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(user)) {
          Value result = broadcastOp.getResult();
          if (!visited.count(result)) {
            visited.insert(result);
            worklist.push_back(result);
          }
        } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(user)) {
          Value result = bitcastOp.getResult();
          if (!visited.count(result)) {
            visited.insert(result);
            worklist.push_back(result);
          }
        }
      }
    }
    return false;
  }

  // 检查在同一 block 中store之后是否有对相同指针的 load
  bool hasSubsequentLoad(triton::StoreOp storeOp, Value storePtr) const {
    for (Operation *nextOp = storeOp->getNextNode(); nextOp; nextOp = nextOp->getNextNode()) {
      if (auto loadOp = dyn_cast<triton::LoadOp>(nextOp)) {
        if (loadOp.getPtr() == storePtr) {
          return true;
        }
      }
      // 如果遇到另一个 store 到同一个指针，停止搜索
      if (auto nextStore = dyn_cast<triton::StoreOp>(nextOp)) {
        if (nextStore.getPtr() == storePtr) {
          break;
        }
      }
    }
    return false;
  }

  LogicalResult matchAndRewrite(triton::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Value storePtr = storeOp.getPtr();
    Value basePtr = getBasePointer(storePtr);
    
    // 检查基础指针是否是保护的参数
    if (isPreservedArgument(basePtr)) {
      // 这个参数的 store 应该被保护
      return failure();
    }
    
    // 检查是否是函数参数
    if (isFunctionArgument(basePtr)) {
      // 只有标记为临时缓冲区的参数才能考虑删除
      if (!isTempBufferArgument(basePtr)) {
        return failure();
      }
      // 继续检查是否是 dead store...
    }
    
    // 检查基础指针在整个函数中是否被 load 过
    if (basePointerEverLoaded(basePtr)) {
      return failure();
    }
    
    // 检查在同一 block 中是否有后续的 load
    if (hasSubsequentLoad(storeOp, storePtr)) {
      return failure();
    }
    
    // 所有检查通过，这是一个 dead store，可以安全删除
    rewriter.eraseOp(storeOp);
    return success();
  }
};

// =========================================================================
// Dead Store Elimination Pass
// =========================================================================
struct TritonDeadStoreEliminationPass 
    : public impl::TritonDeadStoreEliminationBase<TritonDeadStoreEliminationPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 将 preserveArgs 选项转换为 DenseSet
    llvm::SmallDenseSet<int32_t, 16> preserveArgSet;
    for (int32_t arg : preserveArgs) {
      preserveArgSet.insert(arg);
    }

    // 添加 Dead Store Elimination pattern，传入保护的参数集合
    patterns.add<DeadStoreEliminationPattern>(context, preserveArgSet);

    // 使用 GreedyPatternRewriteDriver 应用 pattern
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::triton
