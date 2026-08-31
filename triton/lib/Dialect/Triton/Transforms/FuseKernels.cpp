#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONFUSEKERNELS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// 旧的辅助函数保持不变，用于解析单个 "N->M"
static FailureOr<std::pair<int, int>> parseMapping(StringRef s) {
  auto arrow = s.find("->");
  if (arrow == StringRef::npos)
    return failure();
  int lhs = -1, rhs = -1;
  if (s.take_front(arrow).getAsInteger(10, lhs))
    return failure();
  if (s.drop_front(arrow + 2).getAsInteger(10, rhs))
    return failure();
  return std::make_pair(lhs, rhs);
}

// 新的辅助函数：解析像 "0->2,1->3" 这样的字符串
static FailureOr<SmallVector<std::pair<int, int>>>
parseMappings(StringRef s) {
  SmallVector<std::pair<int, int>> mappings;
  SmallVector<StringRef> parts;
  s.split(parts, ',');

  for (StringRef part : parts) {
    auto mapRes = parseMapping(part.trim());
    if (failed(mapRes))
      return failure();
    mappings.push_back(*mapRes);
  }
  return mappings;
}

// Copy the body of `from` into `into` with provided mapping.
static void cloneBodyInto(FuncOp from, OpBuilder &b, IRMapping &mapper,
                          Block *insertIntoBlock, Block::iterator insertPt) {
  for (Operation &op : from.getBody().front().getOperations()) {
    // Skip the return; we'll manage returns manually.
    if (isa<triton::ReturnOp>(&op))
      continue;
    b.setInsertionPoint(insertIntoBlock, insertPt);
    b.clone(op, mapper);
  }
}

// Keep signature stable; body-only canonicalization is handled by targeted
// IRMapping in the pass (no type-based global merging here).
static void canonicalizeArgsBody(FuncOp) {}


struct TritonFuseKernelsPass : public impl::TritonFuseKernelsBase<TritonFuseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (producer.empty() || consumer.empty()) {
      module.emitError("triton-fuse-kernels requires --producer and --consumer");
      signalPassFailure();
      return;
    }
    // === 修改 1: 使用新的解析器 ===
    auto mappingsRes = parseMappings(mapping);
    if (failed(mappingsRes)) {
      module.emitError("Invalid --mapping format; expected N->M,K->L,...");
      signalPassFailure();
      return;
    }
    const SmallVector<std::pair<int, int>> &parsedMappings = *mappingsRes;

    SymbolTable symTable(module);
    auto prod = symTable.lookup<FuncOp>(producer);
    auto cons = symTable.lookup<FuncOp>(consumer);
    if (!prod || !cons) {
      module.emitError("producer or consumer function not found");
      signalPassFailure();
      return;
    }

    // Create fused function type: union of inputs from prod/cons excluding the
    // consumer arg that will be replaced, and unify the output to match
    // consumer's output signature.
    FunctionType prodTy = dyn_cast<FunctionType>(prod.getFunctionType());
    FunctionType consTy = dyn_cast<FunctionType>(cons.getFunctionType());
    if (!prodTy || !consTy) {
      module.emitError("expected Triton FuncOp with FunctionType");
      signalPassFailure();
      return;
    }

    SmallVector<Type> fusedInputs;
    SmallVector<Type> fusedResults(consTy.getResults().begin(), consTy.getResults().end());

    DenseSet<int> consumerMappedArgs;
    for (const auto &mapPair : parsedMappings) {
      consumerMappedArgs.insert(mapPair.second);
    }

    // Gather inputs: all of producer inputs; all consumer inputs except mapped index.
    fusedInputs.append(prodTy.getInputs().begin(), prodTy.getInputs().end());
    for (int i = 0, e = consTy.getNumInputs(); i < e; ++i) {
      if (consumerMappedArgs.find(i) == consumerMappedArgs.end()) {
        fusedInputs.push_back(consTy.getInput(i));
      }
    }

    // Build fused function symbol name
    std::string fusedName = producer + std::string("_") + consumer;
    OpBuilder b(module.getContext());
    b.setInsertionPointToEnd(module.getBody());
    auto fusedFn = b.create<FuncOp>(module.getLoc(), fusedName,
                                    FunctionType::get(module.getContext(), fusedInputs, fusedResults));
    fusedFn.setVisibility(SymbolTable::Visibility::Public);

    // Build entry block and map arguments
    Block *entry = fusedFn.addEntryBlock();
    IRMapping mapper;

    // Collect producer output arg indices that are being mapped (these are temporary buffers)
    DenseSet<int> producerTempBufferArgs;
    for (const auto &mapPair : parsedMappings) {
      producerTempBufferArgs.insert(mapPair.first);
    }

    // 收集 producer 上已有的 temp_buffer 标记
    DenseSet<int> existingTempBufferArgs;
    for (auto namedAttr : prod->getAttrs()) {
      std::string attrName = namedAttr.getName().str();
      if (attrName.find(".temp_buffer") != std::string::npos) {
        // 提取 "argN.temp_buffer" 中的 N
        size_t argPos = attrName.find("arg");
        size_t dotPos = attrName.find(".");
        if (argPos != std::string::npos && dotPos != std::string::npos) {
          std::string numStr = attrName.substr(argPos + 3, dotPos - argPos - 3);
          int argNum = std::stoi(numStr);
          existingTempBufferArgs.insert(argNum);
        }
      }
    }

    // Map producer args from start of fused args
    for (auto it : llvm::enumerate(prod.getArguments())) {
      BlockArgument fusedArg = entry->getArgument(it.index());
      mapper.map(it.value(), fusedArg);
      
      // Mark temporary buffer arguments with an attribute
      // 包括新标记的和之前已有的（去重）
      if (producerTempBufferArgs.count(it.index()) || existingTempBufferArgs.count(it.index())) {
        fusedArg.getOwner()->getParentOp()->setAttr(
          "arg" + std::to_string(it.index()) + ".temp_buffer",
          b.getUnitAttr()
        );
      }
    }
    // Map consumer args from tail of fused args, skipping the mapped index.
    // For indices i where producer and consumer have the same type at i, map
    // the consumer argument to the producer's BlockArgument inside the body
    // (keeping the extra fused argument unused to preserve signature shape).
    int fusedIdx = prodTy.getNumInputs();
    for (int i = 0, e = cons.getNumArguments(); i < e; ++i) {
      // 如果当前参数是共享的（被映射的），则跳过
      if (consumerMappedArgs.count(i))
        continue;
      // Always use the corresponding tail argument for consumer parameters.
      // Do NOT assume that same index and type implies same semantics; sizes/strides
      // can differ between producer and consumer (e.g., K vs N), leading to bugs.
      mapper.map(cons.getArgument(i), entry->getArgument(fusedIdx));
      ++fusedIdx;
    }

  // Insert producer body first. Clone verbatim (except return).
    b.setInsertionPointToStart(entry);
    cloneBodyInto(prod, b, mapper, entry, entry->begin());

    // Map consumer's mapped pointer argument to the corresponding producer
    // pointer argument per the provided mapping (prodToConsLHS -> prodToConsRHS).
    // This preserves pointer types and avoids type-mismatch issues when cloning
    // consumer ops (e.g., splat/addptr/load chains).
    // === 修改 4: 循环建立所有 "共享" 参数的连接 ===
    for (const auto &mapPair : parsedMappings) {
      int prodIdx = mapPair.first;
      int consIdx = mapPair.second;
      
      if (prodIdx < 0 || prodIdx >= (int)prod.getNumArguments() ||
          consIdx < 0 || consIdx >= (int)cons.getNumArguments()) {
        fusedFn.emitError("mapping index out of range");
        signalPassFailure();
        return;
      }
      mapper.map(cons.getArgument(consIdx), entry->getArgument(prodIdx));
    }

    // Clone consumer body (except its return).
    for (Operation &op : cons.getBody().front().getOperations()) {
      if (isa<triton::ReturnOp>(op))
        break;
      b.setInsertionPointToEnd(entry);
      b.clone(op, mapper);
    }

    auto consTerm = dyn_cast<triton::ReturnOp>(cons.getBody().front().getTerminator());
    if (!consTerm) {
      fusedFn.emitError("consumer must terminate with triton.return");
      signalPassFailure();
      return;
    }

    SmallVector<Value> fusedReturns;
    fusedReturns.reserve(consTerm.getNumOperands());
    for (Value operand : consTerm.getOperands()) {
      Value mapped = mapper.lookupOrNull(operand);
      if (!mapped) {
        fusedFn.emitError("return operand missing after fusion");
        signalPassFailure();
        return;
      }
      fusedReturns.push_back(mapped);
    }

    b.setInsertionPointToEnd(entry);
    b.create<triton::ReturnOp>(consTerm.getLoc(), fusedReturns);

    canonicalizeArgsBody(fusedFn);
    
    // === 清理工作：删除原有的 producer 和 consumer 函数 ===
    // 现在已经创建了融合后的函数，可以安全地删除原来的两个函数
    prod.erase();
    cons.erase();
  }
};

} // namespace
} // namespace mlir::triton
