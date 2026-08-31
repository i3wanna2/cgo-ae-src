#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONFLASHATTENTIONFUSION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// =========================================================================
// Flash Attention 融合优化
// 
// 将两个循环的模式：
//   Loop1: QK, max, sum, store
//   Loop2: load, normalize, dot with V
// 
// 融合为单个循环：
//   Loop1: QK, max, sum, dot with V (累积)
//   (循环外): 除以 sum
// =========================================================================

struct FlashAttentionFusionPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp firstLoop,
                                PatternRewriter &rewriter) const override {
    
    static int attemptNumber = 0;
    attemptNumber++;
    // llvm::errs() << "\n===== DEBUG: FlashAttentionFusion Attempt #" << attemptNumber << " =====\n";
    
    // 1. 匹配第一个循环：应该有 2 个 iter_args (max, sum)
    if (firstLoop.getNumRegionIterArgs() != 2) {
    //   llvm::errs() << "DEBUG: First loop has " << firstLoop.getNumRegionIterArgs() 
    //                << " iter_args, expected 2. Skipping.\n";
      return failure();
    }
    
    // llvm::errs() << "DEBUG: First loop matches (2 iter_args)\n";
    
    // 2. 找到紧随其后的第二个循环
    Operation *nextOp = firstLoop->getNextNode();
    while (nextOp && !isa<scf::ForOp>(nextOp)) {
      nextOp = nextOp->getNextNode();
    }
    
    if (!nextOp) {
    //   llvm::errs() << "DEBUG: No second loop found after first loop\n";
      return failure();
    }
    
    auto secondLoop = dyn_cast<scf::ForOp>(nextOp);
    if (!secondLoop) {
    //   llvm::errs() << "DEBUG: Next operation is not a ForOp\n";
      return failure();
    }
    
    // llvm::errs() << "DEBUG: Found second loop\n";
    
    // 3. 验证第二个循环使用第一个循环的结果
    // max 应该在第二个循环内部使用（用于减法）
    // sum 应该在第二个循环之后使用（用于归一化）
    Value firstMax = firstLoop.getResult(0);
    Value firstSum = firstLoop.getResult(1);
    
    bool usesMax = false;
    for (Operation *user : firstMax.getUsers()) {
      if (secondLoop->isProperAncestor(user)) {
        usesMax = true;
        break;
      }
    }
    
    if (!usesMax) {
    //   llvm::errs() << "DEBUG: Second loop does not use first loop's max result\n";
      return failure();
    }
    
    // llvm::errs() << "DEBUG: Second loop uses first loop's max result\n";
    
    // 4. 在第二个循环中查找关键操作
    Block *secondBody = secondLoop.getBody();
    triton::DotOp dotOp = nullptr;
    triton::LoadOp vLoad = nullptr;
    
    // llvm::errs() << "DEBUG: Searching for DotOp and LoadOp in second loop\n";
    
    for (Operation &op : secondBody->getOperations()) {
      if (auto dot = dyn_cast<triton::DotOp>(&op)) {
        if (!dotOp) {
          dotOp = dot;
        //   llvm::errs() << "DEBUG: Found DotOp in second loop\n";
        }
      }
      // V 的 load 通常是第二个 load（第一个是从 arg2 load QK scores）
      if (auto load = dyn_cast<triton::LoadOp>(&op)) {
        if (vLoad) {
          // 找到第二个 load，这应该是 V
        //   llvm::errs() << "DEBUG: Found second LoadOp (V load)\n";
        } else {
          vLoad = load;
        //   llvm::errs() << "DEBUG: Found first LoadOp\n";
        }
      }
    }
    
    if (!dotOp) {
    //   llvm::errs() << "DEBUG: No DotOp found in second loop\n";
      return failure();
    }
    
    // 5. 找到 V 的 load 操作（dot 的第二个操作数的来源）
    // llvm::errs() << "DEBUG: Finding V load from DotOp operand\n";
    Value dotB = dotOp.getB();  // 使用 getB() 而不是 getRhs()
    auto vLoadOp = dotB.getDefiningOp<triton::LoadOp>();
    if (!vLoadOp) {
    //   llvm::errs() << "DEBUG: DotOp's B operand is not directly a LoadOp, checking for ExtFOp\n";
      // 可能经过了类型转换
      if (auto extOp = dotB.getDefiningOp<arith::ExtFOp>()) {
        // llvm::errs() << "DEBUG: Found ExtFOp, getting its input\n";
        vLoadOp = extOp.getIn().getDefiningOp<triton::LoadOp>();
      }
      if (!vLoadOp) {
        // llvm::errs() << "DEBUG: Could not find V LoadOp\n";
        return failure();
      }
    }
    // llvm::errs() << "DEBUG: Found V LoadOp\n";
    
    // 6. 修改第一个循环，添加第三个 iter_arg 用于累积 dot 结果
    // llvm::errs() << "DEBUG: Starting loop transformation\n";
    Block *firstBody = firstLoop.getBody();
    
    // 获取第一个循环的位置信息
    Location loc = firstLoop.getLoc();
    
    // llvm::errs() << "DEBUG: Creating accumulator initialization\n";
    // 创建新的初始累加值（零张量，与 dot 输出类型相同）
    auto dotType = dotOp.getType();
    // llvm::errs() << "DEBUG: dotType = " << dotType << "\n";
    rewriter.setInsertionPoint(firstLoop);
    auto zeroAttr = rewriter.getZeroAttr(dotType);
    // llvm::errs() << "DEBUG: zeroAttr = " << (zeroAttr ? "valid" : "null") << "\n";
    if (!zeroAttr) {
    //   llvm::errs() << "ERROR: getZeroAttr returned null!\n";
      return failure();
    }
    auto accInit = rewriter.create<arith::ConstantOp>(loc, dotType, zeroAttr);
    // llvm::errs() << "DEBUG: accInit = " << accInit << "\n";
    // llvm::errs() << "DEBUG: Created accumulator initialization\n";
    
    // 创建新的循环，增加一个 iter_arg
    // llvm::errs() << "DEBUG: Creating new loop with additional iter_arg\n";
    SmallVector<Value> newIterArgs;
    newIterArgs.append(firstLoop.getInitArgs().begin(), firstLoop.getInitArgs().end());
    newIterArgs.push_back(accInit);
    
    auto newLoop = rewriter.create<scf::ForOp>(
      loc,
      firstLoop.getLowerBound(),
      firstLoop.getUpperBound(),
      firstLoop.getStep(),
      newIterArgs
    );
    // llvm::errs() << "DEBUG: New loop created\n";
    
    Block *newBody = newLoop.getBody();
    
    // 7. 复制第一个循环的 body 到新循环
    // llvm::errs() << "DEBUG: Cloning first loop body to new loop\n";
    IRMapping mapper;
    for (auto [oldArg, newArg] : llvm::zip(firstBody->getArguments(), newBody->getArguments())) {
      mapper.map(oldArg, newArg);
    }
    
    rewriter.setInsertionPointToStart(newBody);
    for (Operation &op : firstBody->getOperations()) {
      if (!isa<scf::YieldOp>(&op)) {
        rewriter.clone(op, mapper);
      }
    }
    
    // 8. 在新循环中添加 V 的 load 和 dot 操作
    // 首先需要映射 V load 的指针和 mask
    rewriter.setInsertionPointToEnd(newBody);
    
    // 克隆 V load 相关的指针计算和 load 操作
    // 这需要从第二个循环中复制相关操作
    IRMapping vMapper;
    // 复制循环归纳变量
    vMapper.map(secondBody->getArgument(0), newBody->getArgument(0));

    // 首先，映射所有来自循环外部的值（函数参数和循环外定义的值）
    // 加强版：
    //  - 若值定义在 secondLoop 的任意区域内，则视为内部依赖，必须克隆（不在此函数中映射）
    //  - 若值定义在 firstLoop 内部，则使用 firstLoop 克隆映射（mapper）
    //  - 若值定义在 firstLoop 与 secondLoop 之间（同一父 block），则也视为需要克隆
    //  - 仅当值严格支配 firstLoop（位于其之前或祖先区域）时，才允许直接映射到自身
    Operation *firstLoopOp = firstLoop.getOperation();
    Operation *secondLoopOp = secondLoop.getOperation();
    Block *parentBlock = firstLoopOp->getBlock();

    auto mapExternalValues = [&](Value v) {
      if (vMapper.contains(v))
        return; // 避免覆盖已有映射

      if (Operation *defOp = v.getDefiningOp()) {
        // 1) 值定义在第二个循环区域内（包含嵌套）——不在此映射，交由依赖收集进行克隆
        if (secondLoopOp->isProperAncestor(defOp)) {
          return;
        }

        // 2) 值定义在第一个循环区域内 —— 使用 firstLoop 克隆映射
        if (firstLoopOp->isProperAncestor(defOp)) {
          Value mapped = mapper.lookupOrDefault(v);
          if (mapped)
            vMapper.map(v, mapped);
          else
            llvm::errs() << "WARNING: Value inside firstLoop has no mapper mapping: " << v << "\n";
          return;
        }

        // 3) 值定义在与两个循环相同的父 block 中
        if (defOp->getBlock() == parentBlock) {
          if (defOp->isBeforeInBlock(firstLoopOp)) {
            // 定义在第一个循环之前，支配 newLoop，安全地直接使用
            vMapper.map(v, v);
            return;
          } else {
            // 定义在第一个循环之后（两循环之间或之后），不支配 newLoop，需克隆
            return; // 留给依赖收集
          }
        }

        // 4) 其它情况：若 defOp 是 firstLoop 的祖先区域（如函数入口块中）内的 op，视为支配
        if (defOp->getParentOp() && defOp->getParentOp()->isAncestor(firstLoopOp)) {
          vMapper.map(v, v);
          return;
        }

        // 默认：保守起见不直接映射，由依赖收集决定是否克隆
        return;
      }

      // BlockArgument 情况
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        Operation *ownerOp = barg.getOwner() ? barg.getOwner()->getParentOp() : nullptr;
        // 第二个循环体的参数（iv 和 iter_args）不应直接引用到新循环体内
        if (ownerOp == secondLoopOp) {
          // 归纳变量应已单独映射；其它 iter_args 不支持，保守处理：不映射，交由失败路径
          llvm::errs() << "WARNING: Second loop block argument encountered, not mapping directly: " << v << "\n";
          return;
        }
        // 若为包含 firstLoop 的祖先 op（例如函数形参等），可直接使用
        if (ownerOp && ownerOp->isAncestor(firstLoopOp)) {
          vMapper.map(v, v);
          return;
        }
        // 其他情况保守不映射
        return;
      }
    };

    // 复制 V load 之前的所有依赖操作（包括：
    //  - secondLoop 区域内的 op（含嵌套）
    //  - 与两循环同 block 但在 firstLoop 之后定义的 op）
    SmallVector<Operation *> opsToClone;
    DenseSet<Operation *> seen;
    std::function<void(Value)> collectDeps = [&](Value v) {
      if (Operation *defOp = v.getDefiningOp()) {
        bool inSecondLoopRegion = secondLoopOp->isProperAncestor(defOp);
  bool betweenLoopsSameBlock = (defOp->getBlock() == parentBlock) &&
             firstLoopOp->isBeforeInBlock(defOp) &&
             defOp->isBeforeInBlock(secondLoopOp);

        if ((inSecondLoopRegion || betweenLoopsSameBlock) && !seen.contains(defOp)) {
          seen.insert(defOp);
          // 先收集该 op 的依赖
          for (Value operand : defOp->getOperands())
            collectDeps(operand);
          opsToClone.push_back(defOp);
          return;
        }

        // 其余情况尝试作为外部值映射（若不支配，则稍后会被检测/失败）
        mapExternalValues(v);
        return;
      }

      // BlockArgument 或常量等（无定义 op）
      mapExternalValues(v);
    };
    
    collectDeps(vLoadOp.getPtr());
    if (vLoadOp.getMask())
      collectDeps(vLoadOp.getMask());
    

    // 克隆这些操作
    for (unsigned i = 0; i < opsToClone.size(); ++i) {
      Operation *op = opsToClone[i];
      if (!isa<scf::YieldOp>(op) && !isa<triton::LoadOp>(op)) {
        // Validate all operands are mapped
        for (unsigned j = 0; j < op->getNumOperands(); ++j) {
          Value operand = op->getOperand(j);
          Value mapped = vMapper.lookupOrDefault(operand);
          if (!mapped) {
            return failure();
          }
        }
        
        Operation *cloned = rewriter.clone(*op, vMapper);
      }
    }
    
    // 克隆 V load
    auto newVLoad = rewriter.clone(*vLoadOp.getOperation(), vMapper);
    Value vData = newVLoad->getResult(0);
    
    // 9. 在第一个循环中计算 exp(QK - max)
    // 找到第一个循环中存储的 QK scores（store 之前的值）
    triton::StoreOp storeOp = nullptr;
    for (Operation &op : *newBody) {
      if (auto st = dyn_cast<triton::StoreOp>(&op)) {
        storeOp = st;
        break;
      }
    }
    
    if (!storeOp)
      return failure();
    
    // storeOp 是从 newBody 中找到的，它的 value 已经是在 newBody 中的，不需要 lookup
    Value qkScores = storeOp.getValue();
    
    // 10. 计算 exp(QK - max) 并做 dot
    // 找到 yield 中的第一个操作数（max）
    // 需要从原始 firstBody 的 yield 中获取，然后通过 mapper 查找对应的新值
    Value oldMax = firstBody->getTerminator()->getOperand(0);
    Value maxValue = mapper.lookupOrDefault(oldMax);
    
    Value oldMaxInTerminator = firstBody->getTerminator()->getOperand(0);
    Value m_new = mapper.lookupOrDefault(oldMaxInTerminator);
    
    // =====================================================================
    // ======================== 新增代码开始 ===============================
    // =====================================================================

    // A. 获取 m_old, 即迭代开始时的 max 值。它在新循环的 block arugment 中
    //    (arg 0 是归纳变量, arg 1 是 max, arg 2 是 sum)
    Value m_old = newBody->getArgument(1);
    
    // B. 计算缩放因子 scale = exp(m_old - m_new)
    //    这个计算逻辑应该与 sum 的更新逻辑完全一致
    Value maxDiff = rewriter.create<arith::SubFOp>(loc, m_old, m_new);
    Value scale = rewriter.create<math::ExpOp>(loc, maxDiff);

    // C. 获取 acc_old 并对其进行缩放
    Value acc_old = newBody->getArgument(newBody->getNumArguments() - 1);
    //    需要将 scale (通常是 tensor<128x1xf32>) 广播到 acc_old 的形状
    Value scaleBroadcast = rewriter.create<triton::BroadcastOp>(
        loc, acc_old.getType(), scale);
    Value scaledAcc = rewriter.create<arith::MulFOp>(loc, acc_old, scaleBroadcast);

    // 检查关键值
    if (!maxValue || !qkScores) {
      return failure();
    }

    // maxValue 的类型决定了中间计算的类型（通常是 f32）
    auto maxType = cast<RankedTensorType>(maxValue.getType());
    auto qkScoresType = cast<RankedTensorType>(qkScores.getType());
    
    // qkScores 是 f16，需要转换为与 maxValue 相同的元素类型（f32）
    auto qkF32Type = RankedTensorType::get(qkScoresType.getShape(), maxType.getElementType());
    Value qkF32 = rewriter.create<arith::ExtFOp>(loc, qkF32Type, qkScores);

    // broadcast maxValue 到与 qkF32 相同的 shape，类型也应该是 f32
    auto maxBroadcastType = qkF32Type;  // tensor<128x128xf32>
    Value maxBroadcast = rewriter.create<triton::BroadcastOp>(
      loc, maxBroadcastType, maxValue);

    Value qkMinusMax = rewriter.create<arith::SubFOp>(loc, qkF32, maxBroadcast);
    Value expQK = rewriter.create<math::ExpOp>(loc, qkMinusMax);
    
    // 转换为 f16（如果需要，以便与 V 做 dot）
    Value expQKForDot = expQK;
    auto vDataType = cast<RankedTensorType>(vData.getType());
    auto expQKType = cast<RankedTensorType>(expQK.getType());
 
    if (vDataType.getElementType() != expQKType.getElementType()) {
      // 创建与 vData 相同 shape 但元素类型为目标类型的 tensor type
      auto targetType = RankedTensorType::get(
        expQKType.getShape(), vDataType.getElementType());
      expQKForDot = rewriter.create<arith::TruncFOp>(loc, targetType, expQK);
    }
    
    // 执行 dot
    Value accArg = newBody->getArgument(newBody->getNumArguments() - 1);
    Value newAcc = rewriter.create<triton::DotOp>(
      loc, dotType, expQKForDot, vData, scaledAcc);

    // 11. 创建新的 yield，包含原来的 max, sum，以及新的 acc
    // 原始 yield 的操作数需要通过 mapper 查找
    auto firstYield = cast<scf::YieldOp>(firstBody->getTerminator());
    SmallVector<Value> newYieldOperands;

    for (unsigned idx = 0; idx < firstYield.getNumOperands(); ++idx) {
      Value operand = firstYield.getOperand(idx);
      Value mappedValue = mapper.lookupOrDefault(operand);
      if (!mappedValue) {
        return failure();
      }
      newYieldOperands.push_back(mappedValue);
    }
    
    if (!newAcc) {
      return failure();
    }
    newYieldOperands.push_back(newAcc);

    // Validate all yield operands
    for (unsigned i = 0; i < newYieldOperands.size(); ++i) {
      if (!newYieldOperands[i]) {
        return failure();
      }
    }
    
    rewriter.setInsertionPointToEnd(newBody);
    auto newYield = rewriter.create<scf::YieldOp>(loc, newYieldOperands);

    // 12-13. 在循环外做归一化：result / sum
    Value unnormalizedResult = newLoop.getResult(2);  // 新循环的第三个结果是未归一化的累加器
    Value finalSum = newLoop.getResult(1);  // 新循环的第二个结果是 sum
    
    if (!unnormalizedResult || !finalSum) {
        return failure();
    }
    
    rewriter.setInsertionPointAfter(newLoop);
    auto resultType = cast<RankedTensorType>(unnormalizedResult.getType());
    Value sumBroadcast = rewriter.create<triton::BroadcastOp>(
      loc, resultType, finalSum);
    
    if (!sumBroadcast) {
      return failure();
    }
    
    Value normalizedResult = rewriter.create<arith::DivFOp>(
      loc, unnormalizedResult, sumBroadcast);
    
    if (!normalizedResult) {
      return failure();
    }

    // 额外的支配性检查：确保 normalizedResult、sumBroadcast 的操作数定义支配其使用位置
    auto checkDominance = [&](Operation *useOp) -> bool {
      DominanceInfo dom(useOp->getParentOp());
      bool ok = true;
      useOp->walk([&](Operation *inner) {
        for (Value operand : inner->getOperands()) {
          if (Operation *def = operand.getDefiningOp()) {
            if (!dom.properlyDominates(def, inner)) {
              ok = false;
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });
      return ok;
    };

    if (!checkDominance(normalizedResult.getDefiningOp())) {
      return failure();
    }
    
    // 首先，找到并替换使用 secondLoop 结果的 divf 操作
    llvm::errs() << "DEBUG: Finding and replacing existing divf operation\n";
    arith::DivFOp existingDivOp = nullptr;
    if (secondLoop.getNumResults() > 0) {
      for (auto &use : secondLoop.getResult(0).getUses()) {
        if (auto divOp = dyn_cast<arith::DivFOp>(use.getOwner())) {
          existingDivOp = divOp;
          llvm::errs() << "DEBUG: Found existing divf: " << *divOp.getOperation() << "\n";
          break;
        }
      }
    }
    
    if (existingDivOp) {
      rewriter.replaceOp(existingDivOp, normalizedResult);
    } else {
    }
    
    // 现在删除第二个循环（它的结果已经没有用户了）
    rewriter.eraseOp(secondLoop);

    // 验证新循环的结果都是有效的
    for (unsigned i = 0; i < newLoop.getNumResults(); ++i) {
      Value result = newLoop.getResult(i);
      if (!result) {
        return failure();
      }
    }
    
    rewriter.replaceOp(firstLoop, newLoop.getResults().take_front(2));
    return success();
  }
};

// =========================================================================
// Fuse pattern for simple softmax (no max tracking):
// - First loop has 1 iter_arg (sum)
// - Second loop computes exp(x)/sum and dot with V accumulating into a tensor
// Transform:
// - Add an extra iter_arg (accumulator) to the first loop
// - Inside the loop, reuse numerator exp(x) (from first loop) and perform dot with V
// - After the loop, divide accumulator by broadcast(sum)
// =========================================================================
struct FlashAttentionFusionSimplePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp firstLoop,
                                PatternRewriter &rewriter) const override {
    // 1. 基础检查
    if (firstLoop.getNumRegionIterArgs() != 1) return failure();

    Operation *nextOp = firstLoop->getNextNode();
    while (nextOp && !isa<scf::ForOp>(nextOp))
      nextOp = nextOp->getNextNode();
    if (!nextOp) return failure();
    auto secondLoop = dyn_cast<scf::ForOp>(nextOp);
    if (!secondLoop) return failure();

    if (secondLoop.getNumRegionIterArgs() != 1) return failure();

    Value sumVal = firstLoop.getResult(0);
    bool usedInSecond = false;
    for (Operation *user : sumVal.getUsers()) {
      if (secondLoop->isProperAncestor(user)) { usedInSecond = true; break; }
    }
    if (!usedInSecond) return failure();

    // 2. 寻找关键 Op
    Block *secondBody = secondLoop.getBody();
    triton::DotOp dotOp = nullptr;
    for (Operation &op : secondBody->getOperations()) {
      if (auto d = dyn_cast<triton::DotOp>(&op)) { dotOp = d; break; }
    }
    if (!dotOp) return failure();

    // Helper to trace back through trans/extf/truncf to find the LoadOp
    std::function<triton::LoadOp(Value)> getLoadFrom = [&](Value v) -> triton::LoadOp {
      if (auto l = v.getDefiningOp<triton::LoadOp>()) return l;
      if (auto e = v.getDefiningOp<arith::ExtFOp>())
        return getLoadFrom(e.getIn());
      if (auto t = v.getDefiningOp<arith::TruncFOp>())
        return getLoadFrom(t.getIn());
      // Handle tt.trans operation
      if (auto trans = v.getDefiningOp<triton::TransOp>())
        return getLoadFrom(trans.getSrc());
      return nullptr;
    };
    triton::LoadOp vLoadOp = getLoadFrom(dotOp.getB());
    if (!vLoadOp) return failure();

    Block *firstBody = firstLoop.getBody();
    Value numeratorValue = nullptr;
    for (Operation &op : *firstBody) {
      if (auto e = dyn_cast<math::ExpOp>(&op)) { numeratorValue = e.getResult(); break; }
      if (auto e2 = dyn_cast<math::Exp2Op>(&op)) { numeratorValue = e2.getResult(); break; }
    }
    if (!numeratorValue) return failure();

    // 3. 开始构建新 Loop
    Location loc = firstLoop.getLoc();
    auto accType = dotOp.getType();
    rewriter.setInsertionPoint(firstLoop);
    
    auto zeroAttr = rewriter.getZeroAttr(accType);
    if (!zeroAttr) return failure();
    auto accInit = rewriter.create<arith::ConstantOp>(loc, accType, zeroAttr);

    SmallVector<Value> newInitArgs;
    newInitArgs.append(firstLoop.getInitArgs().begin(), firstLoop.getInitArgs().end());
    newInitArgs.push_back(accInit);

    auto newLoop = rewriter.create<scf::ForOp>(loc,
                                               firstLoop.getLowerBound(),
                                               firstLoop.getUpperBound(),
                                               firstLoop.getStep(),
                                               newInitArgs);
    Block *newBody = newLoop.getBody();

    // 4. 克隆第一个 Loop 的 Body
    IRMapping mapper;
    for (auto [oldArg, newArg] : llvm::zip(firstBody->getArguments(), newBody->getArguments()))
      mapper.map(oldArg, newArg);
    
    rewriter.setInsertionPointToStart(newBody);
    for (Operation &op : firstBody->getOperations()) {
      if (!isa<scf::YieldOp>(&op)) {
        Operation *clonedOp = rewriter.clone(op, mapper);
        // 【关键修复 1】手动更新 mapper，让后续 Op 能引用到新克隆的 Op
        for (auto [oldRes, newRes] : llvm::zip(op.getResults(), clonedOp->getResults())) {
            mapper.map(oldRes, newRes);
        }
      }
    }

    // 5. 克隆 V Load 的依赖链
    rewriter.setInsertionPointToEnd(newBody);
    IRMapping vMapper;
    vMapper.map(secondBody->getArgument(0), newBody->getArgument(0)); // Map IV

    Operation *firstLoopOp = firstLoop.getOperation();
    Operation *secondLoopOp = secondLoop.getOperation();
    Block *parentBlock = firstLoopOp->getBlock();

    // 辅助函数：映射外部值
    auto mapExternal = [&](Value v) {
      if (vMapper.contains(v)) return;
      if (Operation *defOp = v.getDefiningOp()) {
        if (secondLoopOp->isProperAncestor(defOp)) return; // 在 Loop 2 内部，等待克隆
        if (firstLoopOp->isProperAncestor(defOp)) { // 在 Loop 1 内部，使用上面克隆的结果
          if (Value mv = mapper.lookupOrNull(v)) vMapper.map(v, mv);
          return;
        }
        if (defOp->getBlock() == parentBlock) {
          if (defOp->isBeforeInBlock(firstLoopOp)) { vMapper.map(v, v); return; }
          return; // Loop 间隙的 Op，等待克隆
        }
        if (defOp->getParentOp() && defOp->getParentOp()->isAncestor(firstLoopOp)) {
          vMapper.map(v, v); return;
        }
        return;
      }
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        Operation *ownerOp = barg.getOwner() ? barg.getOwner()->getParentOp() : nullptr;
        if (ownerOp == secondLoopOp) return;
        if (ownerOp && ownerOp->isAncestor(firstLoopOp)) { vMapper.map(v, v); return; }
        return;
      }
    };

    SmallVector<Operation *> opsToClone;
    DenseSet<Operation *> seen;
    std::function<void(Value)> collect = [&](Value v) {
      if (Operation *defOp = v.getDefiningOp()) {
        bool inSecond = secondLoopOp->isProperAncestor(defOp);
        bool between = (defOp->getBlock() == parentBlock) &&
                       firstLoopOp->isBeforeInBlock(defOp) &&
                       defOp->isBeforeInBlock(secondLoopOp);
        
        if ((inSecond || between) && !seen.contains(defOp)) {
          seen.insert(defOp);
          for (Value o : defOp->getOperands()) collect(o);
          opsToClone.push_back(defOp);
          return;
        }
        mapExternal(v);
        return;
      }
      mapExternal(v);
    };

    collect(vLoadOp.getPtr());
    if (vLoadOp.getMask()) collect(vLoadOp.getMask());
    if (vLoadOp.getOther()) collect(vLoadOp.getOther());

    // Collect the chain from dotOp.getB() back to vLoadOp (e.g., trans, extf, truncf)
    // We need to clone these AFTER vLoadOp, so collect them separately
    Value dotB = dotOp.getB();
    SmallVector<Operation *> postLoadOps; // Ops between vLoad and dotB (e.g., trans)
    DenseSet<Operation *> postLoadSeen;
    std::function<void(Value)> collectPostLoadChain = [&](Value v) {
      Operation *defOp = v.getDefiningOp();
      if (!defOp) return;
      if (defOp == vLoadOp.getOperation()) return; // Stop at load
      if (postLoadSeen.contains(defOp)) return;
      postLoadSeen.insert(defOp);
      for (Value o : defOp->getOperands()) collectPostLoadChain(o);
      postLoadOps.push_back(defOp);
    };
    collectPostLoadChain(dotB);

    Operation *vLoadOperation = vLoadOp.getOperation();
    for (Operation *op : opsToClone) {
      if (isa<scf::YieldOp>(op)) continue;
      // 【关键修复 2】移除了 isa<LoadOp> 的过滤，允许 Index Load 被克隆
      if (op == vLoadOperation) continue;
      
      Operation *clonedOp = rewriter.clone(*op, vMapper);
      // 【关键修复 3】手动更新 vMapper
      for (auto [oldRes, newRes] : llvm::zip(op->getResults(), clonedOp->getResults())) {
          vMapper.map(oldRes, newRes);
      }
    }
    
    // 6. 克隆 V Load
    Operation *newVLoadOp = rewriter.clone(*vLoadOp.getOperation(), vMapper);
    // Map the original vLoad result to the new one
    vMapper.map(vLoadOp.getResult(), newVLoadOp->getResult(0));
    
    // 6.5 克隆 post-load 操作链 (e.g., tt.trans)
    for (Operation *op : postLoadOps) {
      Operation *clonedOp = rewriter.clone(*op, vMapper);
      for (auto [oldRes, newRes] : llvm::zip(op->getResults(), clonedOp->getResults())) {
        vMapper.map(oldRes, newRes);
      }
    }
    
    // Get the final vData - either from the last post-load op or directly from vLoad
    Value vData = vMapper.lookupOrDefault(dotB);

    // 7. 计算 Dot
    Value clonedNumerator = mapper.lookupOrNull(numeratorValue);
    if (!clonedNumerator) return failure(); 

    auto vTy = cast<RankedTensorType>(vData.getType());
    auto numTy = cast<RankedTensorType>(clonedNumerator.getType());
    Value lhs = clonedNumerator;
    if (vTy.getElementType() != numTy.getElementType()) {
      auto targetTy = RankedTensorType::get(numTy.getShape(), vTy.getElementType());
      lhs = rewriter.create<arith::TruncFOp>(loc, targetTy, clonedNumerator);
    }

    Value accArg = newBody->getArgument(newBody->getNumArguments() - 1);
    Value newAcc = rewriter.create<triton::DotOp>(loc, accType, lhs, vData, accArg);

    // 8. 创建 Yield
    auto oldYield = cast<scf::YieldOp>(firstBody->getTerminator());
    SmallVector<Value> yieldOps;
    for (Value opnd : oldYield.getOperands()) {
      Value mv = mapper.lookupOrDefault(opnd);
      if (!mv) return failure();
      yieldOps.push_back(mv);
    }
    yieldOps.push_back(newAcc);
    rewriter.create<scf::YieldOp>(loc, yieldOps);

    // 9. 善后：归一化与替换
    Value accRes = newLoop.getResult(1);
    Value finalSum = newLoop.getResult(0);
    
    rewriter.setInsertionPointAfter(newLoop);
    auto resTy = cast<RankedTensorType>(accRes.getType());
    Value sumBC = rewriter.create<triton::BroadcastOp>(loc, resTy, finalSum);
    Value normRes = rewriter.create<arith::DivFOp>(loc, accRes, sumBC);

    if (!secondLoop.getResult(0).use_empty())
      secondLoop.getResult(0).replaceAllUsesWith(normRes);
    
    // 现在删除 secondLoop 是安全的，因为 newLoop 内部不再引用它
    rewriter.eraseOp(secondLoop);
    rewriter.replaceOp(firstLoop, newLoop.getResult(0));

    return success();
  }
};

struct TritonFlashAttentionFusionPass 
    : public impl::TritonFlashAttentionFusionBase<TritonFlashAttentionFusionPass> {
  
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
//   patterns.add<FlashAttentionFusionPattern>(context);
  patterns.add<FlashAttentionFusionSimplePattern>(context);
    
    // Apply patterns (optional optimization, failures are OK)
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::triton
