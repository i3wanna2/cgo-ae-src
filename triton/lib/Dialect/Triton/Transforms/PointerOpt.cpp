#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>


using namespace mlir;
using namespace mlir::triton;
namespace mlir::triton {
    // 假设 Pass 定义名为 TritonPointerOpt
#define GEN_PASS_DEF_TRITONPOINTEROPT
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// =========================================================================
// 步骤 1: 规范化地址计算的模式
// Pattern to rewrite:
//   %ptr1 = tt.splat %base -> tensor<...x1x...>
//   %ptr2 = tt.addptr %ptr1, %row_offset
//   %ptr3 = tt.broadcast %ptr2 -> tensor<...x...x...>
//   %final_ptr = tt.addptr %ptr3, %col_offset
// into:
//   %bcast_row = tt.broadcast %row_offset -> tensor<...x...x...>
//   %total_offset = arith.addi %bcast_row, %col_offset
//   %splat_base = tt.splat %base -> tensor<...x...x...>
//   %final_ptr = tt.addptr %splat_base, %total_offset
// =========================================================================
struct CanonicalizeAddressCalcPattern : public OpRewritePattern<triton::AddPtrOp> {
  using OpRewritePattern<triton::AddPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AddPtrOp finalAddPtr,
                                PatternRewriter &rewriter) const override {
    // --- 匹配模式 ---
    // %final_ptr = tt.addptr %ptr3, %col_offset
    Value colOffset = finalAddPtr.getOffset();
    auto ptr3 = finalAddPtr.getPtr().getDefiningOp<triton::BroadcastOp>();
    if (!ptr3) return failure();

    // %ptr3 = tt.broadcast %ptr2
    auto ptr2 = ptr3.getSrc().getDefiningOp<triton::AddPtrOp>();
    if (!ptr2) return failure();

    // %ptr2 = tt.addptr %ptr1, %row_offset
    Value rowOffset = ptr2.getOffset();
    auto ptr1 = ptr2.getPtr().getDefiningOp<triton::SplatOp>();
    if (!ptr1) return failure();

    // %ptr1 = tt.splat %base
    Value basePtr = ptr1.getSrc();

    // --- 匹配成功，开始重写 ---
    Location loc = finalAddPtr.getLoc();
    auto finalPtrType = finalAddPtr.getType();
    auto finalOffsetType = colOffset.getType();

    // 1. 创建规范化的总偏移量
    auto bcastRow = rewriter.create<triton::BroadcastOp>(loc, finalOffsetType, rowOffset);
    auto totalOffset = rewriter.create<arith::AddIOp>(loc, bcastRow, colOffset);

    // 2. 创建规范化的基地址指针
    auto splatBase = rewriter.create<triton::SplatOp>(loc, finalPtrType, basePtr);

    // 3. 创建最终的指针
    // 明确传入新操作的返回类型
    auto newFinalPtr = rewriter.create<triton::AddPtrOp>(loc, finalAddPtr.getType(), splatBase, totalOffset);

    // 4. 替换旧操作
    rewriter.replaceOp(finalAddPtr, newFinalPtr.getResult());
    
    return success();
  }
};

// =========================================================================
// 步骤 1.5: 消除重复的指针计算（Pointer Calculation CSE）
// Pattern to eliminate duplicate pointer calculations:
//   %a = arith.muli %x, %y
//   %b = tt.broadcast %a -> tensor<...>
//   %c = arith.addi %b, %offset
//   %ptr1 = tt.addptr %base, %c
//   ...
//   %d = tt.broadcast %a -> tensor<...>  // Duplicate broadcast!
//   %e = arith.addi %d, %offset           // Same computation!
//   %ptr2 = tt.addptr %base, %e           // Redundant pointer!
//
// This pattern identifies when two broadcast->addi->addptr chains
// share the same source value and eliminates the duplicate.
// =========================================================================
struct EliminateDuplicatePointerCalcPattern : public OpRewritePattern<triton::AddPtrOp> {
  using OpRewritePattern<triton::AddPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AddPtrOp addPtr,
                                PatternRewriter &rewriter) const override {
    // 匹配模式：tt.addptr(tt.splat(base), arith.addi(tt.broadcast(src), offset))
    auto splatOp = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splatOp) return failure();

    Value basePtr = splatOp.getSrc();
    Value offsetTensor = addPtr.getOffset();

    auto addiOp = offsetTensor.getDefiningOp<arith::AddIOp>();
    if (!addiOp) return failure();

    // 尝试找到broadcast操作
    Value lhs = addiOp.getLhs();
    Value rhs = addiOp.getRhs();
    
    triton::BroadcastOp bcastOp = nullptr;
    Value otherOperand = nullptr;
    
    if (auto lhsBcast = lhs.getDefiningOp<triton::BroadcastOp>()) {
      bcastOp = lhsBcast;
      otherOperand = rhs;
    } else if (auto rhsBcast = rhs.getDefiningOp<triton::BroadcastOp>()) {
      bcastOp = rhsBcast;
      otherOperand = lhs;
    }
    
    if (!bcastOp) return failure();
    
    Value bcastSrc = bcastOp.getSrc();
    
    // 现在在同一个block中向前搜索，看是否有相同的计算
    Block *currentBlock = addPtr->getBlock();
    for (Operation &op : currentBlock->getOperations()) {
      // 只看当前操作之前的操作
      if (&op == addPtr.getOperation()) break;
      
      auto otherAddPtr = dyn_cast<triton::AddPtrOp>(&op);
      if (!otherAddPtr) continue;
      
      // 检查是否匹配相同的模式
      auto otherSplat = otherAddPtr.getPtr().getDefiningOp<triton::SplatOp>();
      if (!otherSplat || otherSplat.getSrc() != basePtr) continue;
      
      auto otherAddi = otherAddPtr.getOffset().getDefiningOp<arith::AddIOp>();
      if (!otherAddi) continue;
      
      Value otherLhs = otherAddi.getLhs();
      Value otherRhs = otherAddi.getRhs();
      
      // 检查是否有相同的broadcast源和相同的额外偏移
      bool match = false;
      if (auto otherBcast = otherLhs.getDefiningOp<triton::BroadcastOp>()) {
        if (otherBcast.getSrc() == bcastSrc && otherRhs == otherOperand) {
          match = true;
        }
      }
      if (auto otherBcast = otherRhs.getDefiningOp<triton::BroadcastOp>()) {
        if (otherBcast.getSrc() == bcastSrc && otherLhs == otherOperand) {
          match = true;
        }
      }
      
      if (match) {
        // 找到了相同的计算，直接复用
        rewriter.replaceOp(addPtr, otherAddPtr.getResult());
        return success();
      }
    }
    
    return failure();
  }
};

// =========================================================================
// 步骤 2: 执行存储转发和死代码删除的模式
// Pattern to match and eliminate:
//   tt.store %ptr, %val, %mask
//   ... (no other writes to %ptr)
//   %loaded_val = tt.load %ptr, %mask
// 
// 重要安全检查：
// - 如果 store 的基础指针（如函数参数）在后续还被使用，则不能删除 store
//   因为后续的 load 可能通过不同的计算路径访问同一块内存
// =========================================================================
struct StoreLoadForwardingPattern : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  // 提取指针计算链的基础值（追溯到 splat 的源或函数参数）
  Value getBasePointer(Value ptr) const {
    Value current = ptr;
    while (current) {
      if (auto addptrOp = current.getDefiningOp<triton::AddPtrOp>()) {
        current = addptrOp.getPtr();
      } else if (auto splatOp = current.getDefiningOp<triton::SplatOp>()) {
        // 找到 splat 的源，这通常是函数参数或其他基础指针
        return splatOp.getSrc();
      } else if (auto broadcastOp = current.getDefiningOp<triton::BroadcastOp>()) {
        current = broadcastOp.getSrc();
      } else {
        // 到达基础值（函数参数、常量等）
        return current;
      }
    }
    return current;
  }

  // 检查基础指针在给定操作之后是否还有使用
  bool basePointerUsedAfter(Value basePtr, Operation *afterOp) const {
    // 遍历基础指针的所有使用
    for (Operation *user : basePtr.getUsers()) {
      // 跳过 afterOp 本身
      if (user == afterOp) continue;
      
      // 检查使用者的位置
      Block *afterBlock = afterOp->getBlock();
      Block *userBlock = user->getBlock();
      
      if (userBlock == afterBlock) {
        // 同一基本块：检查是否在 afterOp 之后
        Operation *op = afterOp->getNextNode();
        while (op) {
          if (op == user) {
            // 找到一个在 afterOp 之后的使用
            return true;
          }
          op = op->getNextNode();
        }
      } else {
        // 不同基本块：检查是否在后续的循环或其他控制流中
        // 保守策略：假设可能有使用
        Operation *parentOp = afterBlock->getParentOp();
        Operation *userParentOp = userBlock->getParentOp();
        
        // 如果 user 在不同的循环中，肯定算"之后"的使用
        if (parentOp != userParentOp) {
          // 检查 userParentOp 是否在 afterOp 之后
          Operation *op = afterOp->getNextNode();
          while (op) {
            if (op == userParentOp || (user->getParentOp() && op->isProperAncestor(user))) {
              return true;
            }
            op = op->getNextNode();
          }
          // 也检查是否在父操作的后续兄弟节点中
          if (parentOp) {
            Operation *sibling = parentOp->getNextNode();
            while (sibling) {
              if (sibling == userParentOp || sibling->isProperAncestor(user)) {
                return true;
              }
              sibling = sibling->getNextNode();
            }
          }
        }
      }
    }
    return false;
  }
  
  // 检查 store 和 load 之间是否有对同一指针的干扰操作
  bool hasInterferingOps(Value ptr, Operation *storeOp, Operation *loadOp) const {
    for (Operation *op = storeOp->getNextNode(); op && op != loadOp; op = op->getNextNode()) {
      // 检查是否是对同一指针的另一个 store
      if (auto otherStore = dyn_cast<triton::StoreOp>(op)) {
        if (otherStore.getPtr() == ptr) {
          return true;
        }
      }
      // 检查是否是对同一指针的 load
      if (auto otherLoad = dyn_cast<triton::LoadOp>(op)) {
        if (otherLoad.getPtr() == ptr) {
          return true;
        }
      }
    }
    return false;
  }

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    
    Value loadPtr = loadOp.getPtr();
    Value loadMask = loadOp.getMask();
    
    // 提取 load 的基础指针
    Value loadBasePtr = getBasePointer(loadPtr);
    
    // 从 loadOp 开始向前搜索
    for (Operation *prevOp = loadOp->getPrevNode(); prevOp; prevOp = prevOp->getPrevNode()) {
      // 如果遇到一个 store...
      auto storeOp = dyn_cast<triton::StoreOp>(prevOp);
      if (!storeOp) continue;

      Value storePtr = storeOp.getPtr();
      
      // 条件1：指针必须是同一个SSA值 (依赖于规范化步骤和CSE)
      if (storePtr != loadPtr) continue;
      
      // 条件2：Mask必须兼容 (这里简化为必须相同)
      if (storeOp.getMask() != loadMask) continue;
      
      // 条件3：没有中间干扰操作
      if (hasInterferingOps(storePtr, storeOp, loadOp)) continue;
      
      // --- 找到匹配的 store-load 对 ---
      // 只做值转发：将 load 替换为 store 的值
      // 但是 **保留 store 操作**，因为：
      // 1. store 可能写入函数参数（如 arg2），这些数据需要被保留
      // 2. 后续可能有其他代码从内存中读取这些数据
      // 3. 这个优化的目的是消除冗余的 load，而不是删除必要的 store
      rewriter.replaceOp(loadOp, storeOp.getValue());
      
      return success();
    }
    
    return failure();
  }
};

// =========================================================================
// 步骤 3: 删除死存储（Dead Store Elimination）
// 删除从未被 load 的 store 操作
// 
// 安全检查：
// - 如果 store 的基础指针是函数参数，检查是否有 temp_buffer 标记
// - 只有标记为临时缓冲区的参数才能删除 dead store
// - 非参数的 store 如果后续没有 load 也可以删除
// =========================================================================
struct DeadStoreEliminationPattern : public OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

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
// Pass 的主体
// =========================================================================
struct TritonPointerOptPass : public impl::TritonPointerOptBase<TritonPointerOptPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 将我们的四个模式添加到集合中（带调试输出）
    patterns.add<CanonicalizeAddressCalcPattern,
                 EliminateDuplicatePointerCalcPattern,
                 StoreLoadForwardingPattern>(context);

    // 使用GreedyPatternRewriteDriver来应用这些模式，直到IR不再变化

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::triton