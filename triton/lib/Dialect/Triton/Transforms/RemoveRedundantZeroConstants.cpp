// Remove redundant zero constants and generate them on-demand.
//
// This pass identifies dense zero constant tensors (arith.constant dense<0>)
// that occupy large register space and replaces their uses with on-demand
// generation (via arith.constant scalar + tt.splat) to reduce register pressure.
//
// Strategy:
// 1. Identify all dense zero constants
// 2. For each use in arith.select, generate a zero tensor locally
// 3. For scf.for iter_args, generate zero inside loop or at entry
// 4. Delete unused dense zero constants

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton;

#define DEBUG_TYPE "triton-remove-redundant-zero-constants"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREMOVEREDUNDANTZEROCONSTANTS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

/// Check if a dense constant is all zeros
static bool isDenseZeroConstant(arith::ConstantOp op) {
  auto attr = op.getValue();
  if (!attr || !isa<DenseElementsAttr>(attr))
    return false;
  
  DenseFPElementsAttr denseAttr = dyn_cast<DenseFPElementsAttr>(attr);
  if (!denseAttr)
    return false;
  
  // Check if all values are zero
  for (auto val : denseAttr.getValues<APFloat>()) {
    if (!val.isZero())
      return false;
  }
  return true;
}

/// Get the scalar element type and shape from a constant tensor
static std::optional<std::pair<Type, SmallVector<int64_t, 4>>>
getScalarTypeAndShape(arith::ConstantOp op) {
  auto attr = op.getValue();
  if (!attr || !isa<DenseElementsAttr>(attr))
    return std::nullopt;
  
  Type resultType = op.getResult().getType();
  auto tensorType = dyn_cast<RankedTensorType>(resultType);
  if (!tensorType)
    return std::nullopt;
  
  Type elemType = tensorType.getElementType();
  SmallVector<int64_t, 4> shape(tensorType.getShape());
  
  return std::make_pair(elemType, shape);
}

/// Generate a zero tensor locally using scalar constant + splat
static Value generateLocalZeroTensor(Location loc, Type tensorType,
                                     OpBuilder &builder) {
  auto tensorTypeVal = dyn_cast<RankedTensorType>(tensorType);
  if (!tensorTypeVal)
    return nullptr;
  
  Type elemType = tensorTypeVal.getElementType();
  
  // Create scalar zero constant
  Value scalarConst;
  if (isa<Float32Type>(elemType)) {
    scalarConst = builder.create<arith::ConstantOp>(
        loc, elemType, builder.getF32FloatAttr(0.0));
  } else if (isa<Float16Type>(elemType)) {
    scalarConst = builder.create<arith::ConstantOp>(
        loc, elemType, builder.getF16FloatAttr(0.0));
  } else if (isa<Float64Type>(elemType)) {
    scalarConst = builder.create<arith::ConstantOp>(
        loc, elemType, builder.getF64FloatAttr(0.0));
  } else {
    return nullptr; // Unsupported type
  }
  
  // Splat to tensor
  auto splatOp = builder.create<triton::SplatOp>(
      loc, tensorType, scalarConst);
  
  return splatOp.getResult();
}

struct RemoveRedundantZeroConstantsPass
    : public impl::TritonRemoveRedundantZeroConstantsBase<
          RemoveRedundantZeroConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();
    
    // Step 1: Collect all dense zero constants
    SmallVector<arith::ConstantOp, 8> zeroConstants;
    root->walk([&](arith::ConstantOp op) {
      if (isDenseZeroConstant(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Found zero constant: " << op << "\n");
        zeroConstants.push_back(op);
      }
    });
    
    if (zeroConstants.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No zero constants found\n");
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << zeroConstants.size()
                            << " zero constants to remove\n");
    
    // Step 2: For each zero constant, replace its uses
    for (auto zeroCst : zeroConstants) {
      auto resultType = zeroCst.getResult().getType();
      LLVM_DEBUG(llvm::dbgs() << "Processing zero constant with type: "
                              << resultType << "\n");
      
      // Collect all uses first (to avoid iterator invalidation)
      SmallVector<OpOperand *, 16> usesToReplace;
      for (OpOperand &use : zeroCst.getResult().getUses()) {
        usesToReplace.push_back(&use);
      }
      
      LLVM_DEBUG(llvm::dbgs() << "  Found " << usesToReplace.size()
                              << " uses\n");
      
      for (OpOperand *use : usesToReplace) {
        Operation *user = use->getOwner();
        
        // Create builder right after the user to insert replacements nearby
        OpBuilder builder(user);
        builder.setInsertionPoint(user);
        
        // Generate zero tensor locally at the use site
        Value localZero = generateLocalZeroTensor(
            zeroCst.getLoc(), resultType, builder);
        
        if (localZero) {
          LLVM_DEBUG(llvm::dbgs() << "  Replaced use in: " << user->getName()
                                  << "\n");
          use->set(localZero);
        } else {
          LLVM_DEBUG(llvm::dbgs() << "  Failed to generate replacement\n");
        }
      }
    }
    
    // Step 3: Clean up: remove now-unused zero constants
    for (auto zeroCst : zeroConstants) {
      if (zeroCst.getResult().use_empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Erasing unused zero constant: "
                                << zeroCst.getResult().getType() << "\n");
        zeroCst.erase();
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Keeping used zero constant: "
                                << zeroCst.getResult().getType() << "\n");
      }
    }
  }
};

} // namespace

} // namespace mlir::triton
