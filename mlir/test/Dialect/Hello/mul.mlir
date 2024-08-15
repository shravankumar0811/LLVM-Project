// RUN: mlir-opt %s --hello-to-affine | FileCheck %s

module {
  func.func @main(%arg0: tensor<256xf64>, %arg1: tensor<256xf64>) -> tensor<256xf64> {
    %0 = hello.mul %arg0, %arg1 : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xf64>
    %1 = hello.add %0, %arg1 : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xf64>
    return %1 : tensor<256xf64>
  }
}

// CHECK: arith.mulf
// CHECK: arith.addf