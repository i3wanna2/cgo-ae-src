//===----------------------------------------------------------------------===//
// TritonAddDivisibilityAttr: ensure tt.divisibility=16 exists on all tt.func
// arguments. Existing attributes are preserved.
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONADDDIVISIBILITYATTR
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

struct TritonAddDivisibilityAttr
    : public impl::TritonAddDivisibilityAttrBase<TritonAddDivisibilityAttr> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder b(mod.getContext());

    // Integer attr value 16
    auto i32Ty = IntegerType::get(mod.getContext(), 32);
    auto sixteen = IntegerAttr::get(i32Ty, 16);

    mod.walk([&](triton::FuncOp func) {
      // Iterate over all arguments, add attr if missing
      for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
        if (func.getArgAttr(i, "tt.divisibility"))
          continue;
        func.setArgAttr(i, "tt.divisibility", sixteen);
      }
    });
  }
};

} // namespace

} // namespace mlir::triton
