#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONCOMBINETENSORSCALARMULTS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

/// Extract float value from a constant
static std::optional<double> extractFloatConstant(Value val) {
  auto defOp = val.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  // Handle direct constant
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      return floatAttr.getValueAsDouble();
    }
    if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat()) {
        return denseAttr.getSplatValue<FloatAttr>().getValueAsDouble();
      }
    }
  }
  // Handle truncf from f32 to f16
  else if (auto truncOp = dyn_cast<arith::TruncFOp>(defOp)) {
    return extractFloatConstant(truncOp.getOperand());
  }
  // Handle splat
  else if (auto splatOp = dyn_cast<triton::SplatOp>(defOp)) {
    return extractFloatConstant(splatOp.getSrc());
  }

  return std::nullopt;
}

/// Check if a value is a parameter (block argument)
static bool isParameter(Value val) {
  return isa<BlockArgument>(val);
}

/// Get the original parameter if value comes from a parameter
static Value getOriginalParameter(Value val) {
  auto defOp = val.getDefiningOp();
  if (!defOp)
    return isa<BlockArgument>(val) ? val : Value();

  // Handle truncf
  if (auto truncOp = dyn_cast<arith::TruncFOp>(defOp)) {
    return getOriginalParameter(truncOp.getOperand());
  }
  // Handle splat
  if (auto splatOp = dyn_cast<triton::SplatOp>(defOp)) {
    return getOriginalParameter(splatOp.getSrc());
  }

  return Value();
}

/// Pass to combine scalar multiplications:
/// - Before loop: load(query) * %arg16 (scalar)
/// - Inside loop: dot(...) * %cst (constant)
/// Optimization: merge both scalars into a single constant
class CombineTensorScalarMultsPass
    : public impl::TritonCombineTensorScalarMultsBase<
          CombineTensorScalarMultsPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    // Find all scf.for loops
    module.walk([&](scf::ForOp firstForOp) {
      // Step 1: Find multiplication by constant inside the first loop
      // Pattern: %dot_result * %cst_N where %cst_N is a constant tensor
      arith::MulFOp firstLoopConstMulOp = nullptr;
      double firstLoopConstFactor = 1.0;
      Value constValue;  // Store the constant value for later comparison

      firstForOp.walk<mlir::WalkOrder::PreOrder>([&](arith::MulFOp mulOp) {
        if (firstLoopConstMulOp)
          return;

        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        // Try to extract constant from rhs
        auto rhsConst = extractFloatConstant(rhs);
        if (rhsConst && *rhsConst != 1.0) {
          firstLoopConstMulOp = mulOp;
          firstLoopConstFactor = *rhsConst;
          constValue = rhs;
          return;
        }

        // Try to extract constant from lhs
        auto lhsConst = extractFloatConstant(lhs);
        if (lhsConst && *lhsConst != 1.0) {
          firstLoopConstMulOp = mulOp;
          firstLoopConstFactor = *lhsConst;
          constValue = lhs;
        }
      });

      if (!firstLoopConstMulOp || firstLoopConstFactor == 1.0) {
        return;
      }

      // Find the next scf.for operation after the first one
      scf::ForOp secondForOp = nullptr;
      bool foundFirst = false;
      firstForOp->getParentOfType<ModuleOp>().walk([&](scf::ForOp forOp) {
        if (!foundFirst) {
          if (forOp == firstForOp)
            foundFirst = true;
          return;
        }
        if (!secondForOp) {
          secondForOp = forOp;
        }
      });

      if (!secondForOp) {
        return;
      }

      // Check for the same constant multiplication in the second loop
      arith::MulFOp secondLoopConstMulOp = nullptr;
      secondForOp.walk<mlir::WalkOrder::PreOrder>([&](arith::MulFOp mulOp) {
        if (secondLoopConstMulOp)
          return;

        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        
        // Check if either operand is using the same constant as the first loop
        if (lhs == constValue || rhs == constValue) {
          secondLoopConstMulOp = mulOp;
        }
      });

      // If we found a matching multiplication in the second loop, remove it
      if (secondLoopConstMulOp) {
        // Get the non-constant operand from the second loop's multiplication
        Value secondLoopNonConst;
        if (secondLoopConstMulOp.getLhs() == constValue) {
          secondLoopNonConst = secondLoopConstMulOp.getRhs();
        } else {
          secondLoopNonConst = secondLoopConstMulOp.getLhs();
        }

        // Replace the multiplication with just the non-constant operand
        secondLoopConstMulOp.replaceAllUsesWith(secondLoopNonConst);
        secondLoopConstMulOp.erase();
      }

      // Step 2: Find the truncf operation before the loop
      // Pattern: %scalar = arith.truncf %arg16 : f32 to f16
      arith::TruncFOp truncfOp = nullptr;
      
      if (auto block = firstForOp->getBlock()) {
        for (auto &op : *block) {
          if (&op == firstForOp.getOperation())
            break;

          if (auto trunc = dyn_cast<arith::TruncFOp>(&op)) {
            // Check if result type is f16
            if (trunc.getResult().getType().isF16()) {
              truncfOp = trunc;
              break;  // Take the first one
            }
          }
        }
      }

      if (!truncfOp) {
        return;
      }

      // Step 3: Create new f16 constant from the loop constant factor
      // Convert loopConstFactor from f32 to f16
      OpBuilder builder(truncfOp);
      builder.setInsertionPointAfter(truncfOp);

      auto f16Type = builder.getF16Type();
      auto f16ConstValue = static_cast<float>(firstLoopConstFactor);
      
      // Create a scalar f16 constant
      auto f16ConstAttr = builder.getFloatAttr(f16Type, f16ConstValue);
      auto f16ConstOp = builder.create<arith::ConstantOp>(truncfOp.getLoc(), f16ConstAttr);

      // Step 4: Create new mul operation after f16 constant
      // Pattern: %scaled_truncf = arith.mulf %truncf_result, %f16_const
      auto truncfResult = truncfOp.getResult();
      auto newMulOp = builder.create<arith::MulFOp>(truncfOp.getLoc(),
                                                     truncfResult, 
                                                     f16ConstOp.getResult());
      // Step 5: Find the splat operation that uses truncfOp result
      // Pattern: %splat = tt.splat %truncf_result
      triton::SplatOp splatOp = nullptr;
      for (auto user : truncfOp->getUsers()) {
        if (auto splat = dyn_cast<triton::SplatOp>(user)) {
          splatOp = splat;
          break;
        }
      }

      if (!splatOp) {
        return;
      }

      // Step 6: Replace the splat to use newMulOp instead of truncfOp
      // Create new splat with newMulOp result
      builder.setInsertionPoint(splatOp);
      auto newSplatOp = builder.create<triton::SplatOp>(
          splatOp.getLoc(), 
          splatOp.getResult().getType(),
          newMulOp.getResult());

      splatOp.replaceAllUsesWith(newSplatOp.getResult());
      splatOp.erase();

      // Step 7: Replace loop constant mul with its LHS operand (the dot result)
      // Pattern: %70 = arith.mulf %66, %cst_1 -> just use %66
      Value dotResult;
      if (extractFloatConstant(firstLoopConstMulOp.getLhs())) {
        dotResult = firstLoopConstMulOp.getRhs();
      } else {
        dotResult = firstLoopConstMulOp.getLhs();
      }

      firstLoopConstMulOp.replaceAllUsesWith(dotResult);
      firstLoopConstMulOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createCombineTensorScalarMultsPass() {
  return std::make_unique<CombineTensorScalarMultsPass>();
}

} // namespace mlir::triton
