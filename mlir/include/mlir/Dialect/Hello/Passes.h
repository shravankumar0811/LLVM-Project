#ifndef MLIR_DIALECT_HELLO_PASSES_H
#define MLIR_DIALECT_HELLO_PASSES_H 
#include<memory>
#include "mlir/Pass/Pass.h" 
namespace mlir {
namespace hello {
#define GEN_PASS_DECL #include "mlir/Dialect/Hello/Passes.h.inc"
std::unique_ptr<Pass> createHelloToAffinePass();
// std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Hello/Passes.h.inc"
} // namespace hello

} // namespace mlir
#endif // MLIR_HELLO_PASSES_H