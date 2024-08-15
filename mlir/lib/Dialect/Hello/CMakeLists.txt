add_mlir_dialect_library(MLIRHello
    HelloDialect.cpp
    HelloOps.cpp
    HelloToAffine.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Hello

    DEPENDS
    MLIRHelloOpsIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRDLTIDialect
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRInferIntRangeInterface
    MLIRIR
    MLIRMemRefDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms

)