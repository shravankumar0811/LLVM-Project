add_mlir_dialect(HelloOps hello)
add_mlir_doc(HelloDialect HelloDialect Hello/ -gen-dialect-doc)
add_mlir_doc(HelloOps HelloOps Hello/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Hello)
add_public_tablegen_target(MLIRHelloPassIncGen)
add_dependencies(mlir-headers MLIRHelloPassIncGen)
add_mlir_doc(Passes HelloPasses ./ -gen-pass-doc)