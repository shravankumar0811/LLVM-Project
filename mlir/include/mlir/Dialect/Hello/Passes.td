#ifndef MLIR_DIALECT_HELLO_PASSES
#define MLIR_DIALECT_HELLO_PASSES

include "mlir/Pass/PassBase.td"
def HelloToAffine : Pass<"hello-to-affine"> {
  let summary = "Fuse affine loop nests";
  let description = [{
    This pass performs fusion of loop nests using a slicing-based approach }];
  let constructor = "hello::createHelloToAffinePass()";
}
#endif // MLIR_DIALECT_HELLO_PASSES