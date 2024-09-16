// Check tosa pipeline works fine with proper lowering.
//
// RUN:   mlir-opt %s --my-tosa-pass-pipeline | FileCheck %s

module {
  func.func @test_broadcast_scalar(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}

// CHECK: gpu.module
// CHECK: llvm.getelementptr