#ifndef HABANA_MLIR_REGISTRATION_PIPELINES_H
#define HABANA_MLIR_REGISTRATION_PIPELINES_H

#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include <cstdint>

namespace mlir {
void createFSPipeline(OpPassManager &pm);

inline void registerAllPipelines() {
  PassPipelineRegistration<>("my-tosa-pass-pipeline",
                             "FS pipeline for optimization level 3",
                             [](OpPassManager &pm) { createFSPipeline(pm); });
}
} // end namespace mlir

#endif // HABANA_MLIR_REGISTRATION_PIPELINES_H