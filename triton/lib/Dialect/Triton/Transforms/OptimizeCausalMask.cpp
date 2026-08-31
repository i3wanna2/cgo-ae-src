#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton;

#define DEBUG_TYPE "optimize-causal-mask"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONOPTIMIZECAUSALMASK
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
std::string direction;
int maskArgIdx;   // mask argument 的索引
int pidAxis;      // 用于计算 rowStart 的 program id axis
// Helper: 追溯指针到原始 block argument
static BlockArgument tracePtrToBlockArg(Value val) {
  // 追溯所有的 addptr 和 splat 操作
  while (val) {
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      return blockArg;
    }

    if (auto addPtr = val.getDefiningOp<triton::AddPtrOp>()) {
      val = addPtr.getPtr();
      continue;
    }

    if (auto splat = val.getDefiningOp<triton::SplatOp>()) {
      val = splat.getSrc();
      continue;
    }

    // 无法继续追溯
    break;
  }

  return nullptr;
}

// Helper: 检查是否是 mask argument（通过参数位置或名称）
static bool isMaskArgument(BlockArgument arg) {
  // 检查参数编号：使用 pass 参数指定的 mask argument 索引
  if (static_cast<int>(arg.getArgNumber()) == maskArgIdx) {
    return true;
  }

  // 备选：检查父函数是否有标记 mask 参数的属性
  auto func = dyn_cast<triton::FuncOp>(arg.getOwner()->getParentOp());
  if (func) {
    std::string attrName = "arg" + std::to_string(arg.getArgNumber()) + ".mask";
    if (func->hasAttr(attrName)) {
      return true;
    }
  }

  return false;
}

// Helper: 在函数中查找已经计算好的全局行索引
// 模式：%globalRows = arith.addi %rowStartSplat, %rowOffsets
static Value findGlobalRowIndices(triton::FuncOp func, int64_t expectedSize) {
  // 在函数开始处查找计算全局行索引的模式
  for (auto &op : func.getBody().front()) {
    if (auto addOp = dyn_cast<arith::AddIOp>(&op)) {
      auto resultType = dyn_cast<RankedTensorType>(addOp.getResult().getType());
      if (!resultType || resultType.getRank() != 1)
        continue;

      if (resultType.getShape()[0] == expectedSize) {
        // 检查是否是 splat + make_range 的模式
        // 这通常是全局索引的计算方式
        return addOp.getResult();
      }
    }
  }
  return nullptr;
}

// Helper: 在循环内查找全局列索引
// 模式：%globalCols = arith.addi %colBaseSplat, %colOffsets
static Value findGlobalColIndices(scf::ForOp forOp, int64_t expectedSize) {
  for (auto &op : forOp.getBody()->getOperations()) {
    if (auto addOp = dyn_cast<arith::AddIOp>(&op)) {
      auto resultType = dyn_cast<RankedTensorType>(addOp.getResult().getType());
      if (!resultType || resultType.getRank() != 1)
        continue;

      if (resultType.getShape()[0] == expectedSize) {
        // 检查是否涉及循环归纳变量
        // 简化实现：返回第一个匹配的
        return addOp.getResult();
      }
    }
  }
  return nullptr;
}

// Helper: 获取当前 block 的行起始位置
// 模式1 (grid = M/BLOCK_M): %rowStart = arith.muli %pid, %blockSize，直接返回
// 模式2 (grid = M): 需要手动计算对齐 rowStart = (pid / BLOCK_M) * BLOCK_M
// 返回 {rowStart, needsAlignment}
// 如果 needsAlignment 为 true，调用者需要自己计算对齐
static std::pair<Value, bool> findRowStartOffset(triton::FuncOp func, int targetAxis) {
  // 首先尝试找 pid * BLOCK_M 的模式 (模式1: grid = M/BLOCK_M)
  for (auto &op : func.getBody().front()) {
    if (auto mulOp = dyn_cast<arith::MulIOp>(&op)) {
      auto lhs = mulOp.getLhs();
      auto rhs = mulOp.getRhs();

      if (auto pidOp = lhs.getDefiningOp<triton::GetProgramIdOp>()) {
        if (pidOp.getAxisAsInt() == targetAxis) {
          // 检查 rhs 是否是常量（BLOCK_M）
          if (rhs.getDefiningOp<arith::ConstantIntOp>() ||
              rhs.getDefiningOp<arith::ConstantOp>()) {
            return {mulOp.getResult(), false};  // 不需要对齐
          }
        }
      }
      if (auto pidOp = rhs.getDefiningOp<triton::GetProgramIdOp>()) {
        if (pidOp.getAxisAsInt() == targetAxis) {
          // 检查 lhs 是否是常量（BLOCK_M）
          if (lhs.getDefiningOp<arith::ConstantIntOp>() ||
              lhs.getDefiningOp<arith::ConstantOp>()) {
            return {mulOp.getResult(), false};  // 不需要对齐
          }
        }
      }
    }
  }
  
  // 找不到 pid * BLOCK_M 模式，返回 pid 本身，需要调用者计算对齐 (模式2: grid = M)
  for (auto &op : func.getBody().front()) {
    if (auto pidOp = dyn_cast<triton::GetProgramIdOp>(&op)) {
      if (pidOp.getAxisAsInt() == targetAxis) {
        return {pidOp.getResult(), true};  // 需要对齐
      }
    }
  }
  
  return {nullptr, false};
}

// Pattern: 优化 causal attention 的循环上界
struct OptimizeCausalLoopBound : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // 避免重复应用导致不收敛：如果已经优化过则跳过
    if (forOp->hasAttr("triton.causal_loop_optimized")) {
      return failure();
    }

    // 1. 检查循环内是否有 causal mask 相关的操作
    // 1. 检查循环内是否有从 mask argument (arg17) 加载的操作
    bool hasMaskLoad = false;
    triton::LoadOp maskLoadOp = nullptr;
    // 遍历循环体内的所有操作
    forOp.getBody()->walk([&](triton::LoadOp loadOp) {
      // 追溯 load 的指针来源
      auto blockArg = tracePtrToBlockArg(loadOp.getPtr());

      // 检查来源是否是 mask argument
      if (blockArg && isMaskArgument(blockArg)) {
        hasMaskLoad = true;
        maskLoadOp = loadOp;            // 保存找到的操作
        return WalkResult::interrupt(); // 找到一个即可，停止遍历
      }
      return WalkResult::advance(); // 继续遍历
    });

    if (!hasMaskLoad) {
    //   printf("Loop does not contain a mask load, skipping optimization.\n");
      return failure();
    }

    // 2. 获取函数和行起始位置
    auto func = forOp->getParentOfType<triton::FuncOp>();
    if (!func) {
      return failure();
    }

    // 3. 获取循环步长
    // 从 maskLoadOp 的结果类型获取期望的 BLOCK_M 大小
    auto loadResultType =
        dyn_cast<RankedTensorType>(maskLoadOp.getResult().getType());
    if (!loadResultType || loadResultType.getRank() < 1) {
      LLVM_DEBUG(llvm::dbgs() << "Mask load result is not a ranked tensor.\n");
      return failure();
    }

    auto lowerbound = forOp.getLowerBound();
    auto upperbound = forOp.getUpperBound();
    // 4. 计算新的上界/下界
    // 对于 causal mask，只需要处理到当前 block 的最后一行
    Location loc = forOp.getLoc();
    rewriter.setInsertionPoint(forOp);
    
    if (direction == "1") {
      // direction == "1": 优化循环下界 (loopN 方向)
      int64_t expectedBlockM = loadResultType.getShape()[1];
      
      // 查找 rowStart，使用 pass 参数指定的 pidAxis
      auto [rowStartVal, needsAlignment] = findRowStartOffset(func, pidAxis);
      if (!rowStartVal) {
        LLVM_DEBUG(llvm::dbgs() << "Could not find row start offset\n");
        return failure();
      }
      
      Value rowStart = rowStartVal;
      auto blockMConst = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), expectedBlockM);
      
      // 如果需要对齐（模式2: grid = M）
      if (needsAlignment) {
        Value groupId = rewriter.create<arith::DivUIOp>(loc, rowStart, blockMConst);
        rowStart = rewriter.create<arith::MulIOp>(loc, groupId, blockMConst);
      }

      // 计算新下界：max(original_lower_bound, rowStart - BLOCK_M)
      auto newLowerBound =
          rewriter.create<arith::SubIOp>(loc, rowStart, blockMConst);

      Value originalLowerBound = forOp.getLowerBound();
      lowerbound = rewriter.create<arith::MaxSIOp>(loc, originalLowerBound,
                                                   newLowerBound);
      Value step = forOp.getStep();
      Value div = rewriter.create<arith::DivSIOp>(loc, lowerbound, step);
      lowerbound = rewriter.create<arith::MulIOp>(loc, div, step);
    } else {
      // direction == "0": 优化循环上界 (loopM 方向)
      int64_t expectedBlockM = loadResultType.getShape()[0];
      
      // 查找 rowStart，使用 pass 参数指定的 pidAxis
      auto [rowStartVal, needsAlignment] = findRowStartOffset(func, pidAxis);
      if (!rowStartVal) {
        LLVM_DEBUG(llvm::dbgs() << "Could not find row start offset\n");
        return failure();
      }
      
      Value rowStart = rowStartVal;
      auto blockMConst = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), expectedBlockM);
      
      // 如果需要对齐（模式2: grid = M）
      if (needsAlignment) {
        Value groupId = rewriter.create<arith::DivUIOp>(loc, rowStart, blockMConst);
        rowStart = rewriter.create<arith::MulIOp>(loc, groupId, blockMConst);
      }

      // 计算新上界：min(original_upper_bound, rowStart + BLOCK_M)
      auto newUpperBound =
          rewriter.create<arith::AddIOp>(loc, rowStart, blockMConst);

      Value originalUpperBound = forOp.getUpperBound();
    //   upperbound = rewriter.create<arith::MinSIOp>(loc, originalUpperBound,
    //                                                newUpperBound);
      Value step = forOp.getStep();
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      Value stepMinusOne = rewriter.create<arith::SubIOp>(loc, step, one);
      Value addInput = rewriter.create<arith::AddIOp>(loc, newUpperBound, stepMinusOne);

      Value div = rewriter.create<arith::DivSIOp>(loc, addInput, step);
      upperbound = rewriter.create<arith::MulIOp>(loc, div, step);
      upperbound = rewriter.create<arith::MinSIOp>(loc, originalUpperBound, upperbound);
    }

    // 5. 传播循环边界 (收集所有需要一起优化的循环)
    llvm::DenseSet<Value> taintedBuffers;
    SmallVector<scf::ForOp> worklist;
    worklist.push_back(forOp);
    llvm::DenseSet<Operation *> loopsToOptimize;
    loopsToOptimize.insert(forOp);

    // 收集所有依赖循环
    for (size_t i = 0; i < worklist.size(); ++i) {
      scf::ForOp currentLoop = worklist[i];
      
      // 跟踪所有写入的缓冲区
      currentLoop.getBody()->walk([&](triton::StoreOp storeOp) {
        if (auto arg = tracePtrToBlockArg(storeOp.getPtr())) {
          taintedBuffers.insert(arg);
        }
      });

      // 查找使用这些缓冲区的后续循环
      Operation *nextOp = currentLoop->getNextNode();
      while (nextOp) {
        if (auto consumerLoop = dyn_cast<scf::ForOp>(nextOp)) {
          if (consumerLoop.getUpperBound() == forOp.getUpperBound()) {
            bool isConsumer = false;
            consumerLoop.getBody()->walk([&](triton::LoadOp loadOp) {
              if (auto arg = tracePtrToBlockArg(loadOp.getPtr())) {
                if (taintedBuffers.contains(arg)) {
                  isConsumer = true;
                  return WalkResult::interrupt();
                }
              }
              return WalkResult::advance();
            });
            if (isConsumer && !loopsToOptimize.contains(consumerLoop)) {
              worklist.push_back(consumerLoop);
              loopsToOptimize.insert(consumerLoop);
            }
          }
        }
        nextOp = nextOp->getNextNode();
      }
    }

    // 6. 按序优化所有相关循环
    IRMapping mapper;
    SmallVector<scf::ForOp> newLoops;

    // 创建所有新循环
    for (auto loop : worklist) {
      rewriter.setInsertionPoint(loop);

      auto newLoop = rewriter.create<scf::ForOp>(
          loc, lowerbound, upperbound, loop.getStep(), loop.getInitArgs());

      for (auto [oldResult, newResult] :
           llvm::zip(loop.getResults(), newLoop.getResults())) {
        mapper.map(oldResult, newResult);
      }

      for (auto [oldArg, newArg] :
           llvm::zip(loop.getBody()->getArguments(),
                     newLoop.getBody()->getArguments())) {
        mapper.map(oldArg, newArg);
      }

      newLoop->setAttr("triton.causal_loop_optimized", rewriter.getUnitAttr());
      newLoops.push_back(newLoop);
    }

    // 克隆并重写循环体
    for (auto [oldLoop, newLoop] : llvm::zip(worklist, newLoops)) {
      Block *oldBlock = oldLoop.getBody();
      Block *newBlock = newLoop.getBody();

      // --- ADD THIS BLOCK ---
      // Fix: scf::ForOp builder usually creates a default terminator. 
      // We must remove it before adding our own to avoid "scf.yield must be last" error.
      if (!newBlock->empty() && newBlock->back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.eraseOp(&newBlock->back());
      }

      rewriter.setInsertionPointToStart(newBlock);

      // 克隆非终结符操作
      for (auto &op : oldBlock->without_terminator()) {
        // 使用 OperationState 创建操作克隆
        OperationState state(op.getLoc(), op.getName().getStringRef());

        // 添加操作数和结果类型
        state.addOperands(
            llvm::to_vector<4>(llvm::map_range(op.getOperands(), [&](Value v) {
              return mapper.lookupOrDefault(v);
            })));
        state.addTypes(op.getResultTypes());

        // 复制属性
        state.attributes = op.getAttrDictionary();

        // 复制区域
        for (Region &r : op.getRegions()) {
          state.addRegion();
        }

        // 创建并插入新操作
        Operation *newOp = rewriter.create(state);

        // 复制区域内容（如果有）
        for (auto [oldRegion, newRegion] :
             llvm::zip(op.getRegions(), newOp->getRegions())) {
          rewriter.cloneRegionBefore(oldRegion, newRegion, newRegion.end(),
                                     mapper);
        }

        // 更新值映射
        mapper.map(op.getResults(), newOp->getResults());
      }

      // 单独处理终结符（yield）
      auto yieldOp = cast<scf::YieldOp>(oldBlock->getTerminator());
      SmallVector<Value, 4> mappedResults;
      for (auto operand : yieldOp.getOperands()) {
        mappedResults.push_back(mapper.lookupOrDefault(operand));
      }

      rewriter.create<scf::YieldOp>(yieldOp.getLoc(), mappedResults);
    }

    // 重定向外部使用
    for (auto [oldLoop, newLoop] : llvm::zip(worklist, newLoops)) {
      for (auto [oldResult, newResult] :
           llvm::zip(oldLoop.getResults(), newLoop.getResults())) {
        for (OpOperand &use : llvm::make_early_inc_range(oldResult.getUses())) {
          Operation *user = use.getOwner();
          bool isInternalUser = false;
          for (auto opToOptimize : loopsToOptimize) {
            if (opToOptimize->isProperAncestor(user)) {
              isInternalUser = true;
              break;
            }
          }
          if (!isInternalUser) {
            rewriter.modifyOpInPlace(user, [&]() { use.set(newResult); });
          }
        }
      }
    }

    // 按相反顺序删除旧循环
    for (auto it = worklist.rbegin(); it != worklist.rend(); ++it) {
      rewriter.eraseOp(*it);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Successfully optimized " << loopsToOptimize.size()
               << " causal loops bounds\n");

    return success();
  }
};

// Pattern: 替换 causal mask load 为动态生成
struct ReplaceCausalMaskLoad : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {

    // 1. 检查 load 的结果类型
    auto resultType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
    if (!resultType || resultType.getRank() != 2) {
      return failure();
    }

    auto shape = resultType.getShape();
    int64_t loadBlockM = shape[0];
    int64_t loadBlockN = shape[1];

    // 2. 检查是否是 mask load（基于指针来源）
    auto blockArg = tracePtrToBlockArg(loadOp.getPtr());
    if (!blockArg || !isMaskArgument(blockArg)) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Found mask load: " << loadOp << " with shape ["
                            << loadBlockM << "x" << loadBlockN << "]\n");

    // 3. 检查是否在 scf.for 循环内
    auto forOp = loadOp->getParentOfType<scf::ForOp>();
    if (!forOp) {
      LLVM_DEBUG(llvm::dbgs() << "Load not in scf.for loop\n");
      return failure();
    }

    // 4. 获取函数
    auto func = loadOp->getParentOfType<triton::FuncOp>();
    if (!func) {
      return failure();
    }

    Location loc = loadOp.getLoc();
    OpBuilder builder(loadOp);

    // 5. 检测 grid 模式并获取正确的 rowStart
    auto [rowStartVal, needsAlignment] = findRowStartOffset(func, pidAxis);
    if (!rowStartVal) {
      LLVM_DEBUG(llvm::dbgs() << "Could not find row start offset\n");
      return failure();
    }

    // 6. 根据 grid 模式生成行索引和列索引
    Value globalRows;
    int64_t blockSize = (direction == "1") ? loadBlockN : loadBlockM;
    int64_t colSize = (direction == "1") ? loadBlockM : loadBlockN;
    Value loopIV = forOp.getInductionVar();
    
    if (needsAlignment) {
      // grid = M 模式: pid 直接是行号
      // 每个 program 处理单个 M 位置，行索引全部相同 = pid
      // globalRows = splat(pid) -> [blockSize] 全是相同值
      globalRows = builder.create<triton::SplatOp>(
          loc, RankedTensorType::get({blockSize}, builder.getI32Type()),
          rowStartVal);
    } else {
      // grid = M/BLOCK_M 模式: rowStart = pid * BLOCK_M
      // globalRows = rowStart + range(0, BLOCK_M)
      auto rowOffsets = builder.create<triton::MakeRangeOp>(
          loc, RankedTensorType::get({blockSize}, builder.getI32Type()),
          0, blockSize);
      auto rowStartSplat = builder.create<triton::SplatOp>(
          loc, RankedTensorType::get({blockSize}, builder.getI32Type()),
          rowStartVal);
      globalRows = builder.create<arith::AddIOp>(loc, rowStartSplat, rowOffsets);
    }

    // 7. 创建全局列索引: loopIV + make_range(0, BLOCK_N)
    // 从循环归纳变量获取列起始位置
    auto colOffsets = builder.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({colSize}, builder.getI32Type()),
        0, colSize);
    auto colStartSplat = builder.create<triton::SplatOp>(
        loc, RankedTensorType::get({colSize}, builder.getI32Type()),
        loopIV);
    Value globalCols = builder.create<arith::AddIOp>(loc, colStartSplat, colOffsets);

    // 8. 根据 direction 调整行列
    if (direction == "1") {
      std::swap(globalCols, globalRows);
      std::swap(blockSize, colSize);
    }

    // 9. 生成 causal mask
    // 扩展维度以进行广播比较
    // global_rows: [loadBlockM] -> [loadBlockM, 1]
    auto globalRowsExpanded = builder.create<triton::ExpandDimsOp>(
        loc, RankedTensorType::get({loadBlockM, 1}, builder.getI32Type()),
        globalRows, 1);

    // global_cols: [loadBlockN] -> [1, loadBlockN]
    auto globalColsExpanded = builder.create<triton::ExpandDimsOp>(
        loc, RankedTensorType::get({1, loadBlockN}, builder.getI32Type()),
        globalCols, 0);

    // 广播到 [loadBlockM, loadBlockN]
    auto globalRowsBroadcast = builder.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get({loadBlockM, loadBlockN}, builder.getI32Type()),
        globalRowsExpanded);

    auto globalColsBroadcast = builder.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get({loadBlockM, loadBlockN}, builder.getI32Type()),
        globalColsExpanded);

    // 生成 causal 条件: mask = (row < col)
    // 对于上三角为 -inf 的 mask，我们需要 row < col 的位置设为 -inf
    auto causalCond =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                      globalRowsBroadcast, globalColsBroadcast);

    // 创建常量：-inf 和 0
    auto elementType = resultType.getElementType();
    Value negInf, zero;

    if (auto fpType = dyn_cast<FloatType>(elementType)) {
      APFloat negInfVal =
          APFloat::getInf(fpType.getFloatSemantics(), /*negative=*/true);
      APFloat zeroVal =
          APFloat::getZero(fpType.getFloatSemantics(), /*negative=*/false);

      // 注意：ConstantFloatOp::build 的参数顺序是 (builder, state, type, value)
      negInf = builder.create<arith::ConstantFloatOp>(loc, fpType, negInfVal);
      zero = builder.create<arith::ConstantFloatOp>(loc, fpType, zeroVal);
    } else {
      return failure();
    }

    // Splat 到张量
    auto negInfTensor =
        builder.create<triton::SplatOp>(loc, resultType, negInf);
    auto zeroTensor = builder.create<triton::SplatOp>(loc, resultType, zero);

    // select: result = causalCond ? -inf : 0
    auto causalMask = builder.create<arith::SelectOp>(loc, causalCond,
                                                      negInfTensor, zeroTensor);

    // 11. 替换原来的 load
    rewriter.replaceOp(loadOp, causalMask.getResult());

    LLVM_DEBUG(llvm::dbgs()
               << "Successfully replaced mask load with dynamic causal mask\n");

    return success();
  }
};

struct TritonOptimizeCausalMaskPass
    : public impl::TritonOptimizeCausalMaskBase<TritonOptimizeCausalMaskPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    GreedyRewriteConfig config; // 可以复用同一个 config
    direction = producer.front();
    maskArgIdx = maskArgIndex;  // 从 pass 参数读取 mask argument 索引
    pidAxis = targetAxis;       // 从 pass 参数读取 target axis
    // llvm::errs() << "direction : "<< direction << ", maskArgIndex: " << maskArgIdx << ", targetAxis: " << pidAxis << "\n";
    // 若 direction 为 "2"，直接退出，不执行任何 pattern
    if (direction == "2") {
      LLVM_DEBUG(llvm::dbgs() << "Skip OptimizeCausalMask pass since direction==2\n");
      return;
    }
    // --- 阶段 1: 仅优化循环边界 ---
    // 创建一个只包含 LoopBound Pattern 的集合
    RewritePatternSet loopPatterns(context);
    loopPatterns.add<OptimizeCausalLoopBound>(context);

    // 首先应用这个 Pattern。
    // 此时所有的 triton::LoadOp 都还存在，LoopBound Pattern 可以成功匹配。
    if (failed(applyPatternsGreedily(getOperation(), std::move(loopPatterns),
                                     config))) {
      signalPassFailure();
      return; // 如果第一阶段失败，则终止
    }

    // --- 阶段 2: 替换 Mask Load ---
    // 注意：这个 pattern 只适用于标准的 causal mask (row < col)
    // 对于 sparse attention 等使用动态 mask 的情况，不应该启用此 pattern
    RewritePatternSet loadPatterns(context);
    loadPatterns.add<ReplaceCausalMaskLoad>(context);

    // 在循环边界优化完成后，现在可以安全地替换掉 Load 了。
    // 这时即使 "证据" (LoadOp) 被删除，也无所谓了，因为 Pattern 1
    // 已经运行过了。
    if (failed(applyPatternsGreedily(getOperation(), std::move(loadPatterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::triton
