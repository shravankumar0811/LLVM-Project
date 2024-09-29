add_mlir_dialect_library(MLIRAffineTransforms
  AffineDataCopyGeneration.cpp
  AffineExpandIndexOps.cpp
  AffineLoopInvariantCodeMotion.cpp
  AffineLoopInvariantHoist.cpp
  AffineLoopPermute.cpp
  AffineLoopSwap.cpp
  AffineLoopMerge.cpp
  AffineLoopUnfold.cpp
  AffineLoopUnrollAndFuse.cpp
  AffineLoopBlocking.cpp
  AffineLoopNormalize.cpp
  AffineParallelize.cpp
  AffineScalarReplacement.cpp
  DecomposeAffineOps.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopTiling.cpp
  LoopUnroll.cpp
  LoopUnrollAndJam.cpp
  PipelineDataTransfer.cpp
  ReifyValueBounds.cpp
  SuperVectorize.cpp
  SimplifyAffineStructures.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen
  MLIRAffinePassIncGen
  MLIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFUtils
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
  MLIRVectorUtils
  MLIRVectorToLLVM
  )

