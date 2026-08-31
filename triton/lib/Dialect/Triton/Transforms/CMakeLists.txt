set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

add_triton_library(TritonTransforms
  Combine.cpp
  LoopAwareCSE.cpp
  LoopInvariantCodeMotion.cpp
  LoopPeeling.cpp
  LoopUnroll.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp
  RewriteTensorDescriptorToPointer.cpp
  ArithTypeConversion.cpp
  FunctionTypeConversion.cpp
  FuseKernels.cpp
  OptimizeFusedLoops.cpp
  PointerOpt.cpp
  DeadStoreElimination.cpp
  RedundantLoadElimination.cpp
  FuseSoftmaxGemm.cpp
  FlashAttentionFusion.cpp
  SoftmaxMicroOpts.cpp
  HoistInvariantLoads.cpp
  EliminateCastRoundtrip.cpp
  PreScaleQuery.cpp
  OptimizeCausalMask.cpp
  FuseMaskIntoDot.cpp
  EarlyLoadV.cpp
  RemoveRedundantZeroConstants.cpp
  AddDivisibilityAttr.cpp
  CombineScalarMults.cpp
  
  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  TritonIR
)
