// RUN: mlir-opt %s --hello-to-affine | FileCheck %s

module {
  func.func @main(%arg0: tensor<256xf64>, %arg1: tensor<256xf64>) -> tensor<256xf64> {
    %0 = hello.add %arg0, %arg1 : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xf64>
    return %0 : tensor<256xf64>
  }
}

// CHECK: arith.addf