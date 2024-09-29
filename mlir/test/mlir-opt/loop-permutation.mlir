// RUN: mlir-opt %s -affine-loop-permute="permutation-map=1,2,0" | FileCheck %s --check-prefix=CHECK-120

// RUN: mlir-opt %s -affine-loop-permute="permutation-map=1,0,2" | FileCheck %s --check-prefix=CHECK-102

// RUN: mlir-opt %s -affine-loop-permute="permutation-map=0,1,2" | FileCheck %s --check-prefix=CHECK-012

// RUN: mlir-opt %s -affine-loop-permute="permutation-map=0,2,1" | FileCheck %s --check-prefix=CHECK-021

// RUN: mlir-opt %s -affine-loop-permute="permutation-map=2,0,1" | FileCheck %s --check-prefix=CHECK-201

// RUN: mlir-opt %s -affine-loop-permute="permutation-map=2,1,0" | FileCheck %s --check-prefix=CHECK-210
module {
  func.func @matrix_multiply(%A: memref<10x10x10xf32>, %B: memref<10x10x10xf32>, %C: memref<10x10x10xf32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        affine.for %k = 0 to 30 {
          %a = affine.load %A[%i, %j, %k] : memref<10x10x10xf32>
          %b = affine.load %B[%i, %j, %k] : memref<10x10x10xf32>
          %c = arith.addf %a, %b : f32
          affine.store %c, %C[%i, %j, %k] : memref<10x10x10xf32>
        }
      }
    }
    return
  }
}

// CHECK-120: affine.for %arg3 = 0 to 20
// CHECK-120:   affine.for %arg4 = 0 to 30
// CHECK-120:     affine.for %arg5 = 0 to 10
// CHECK-120:        affine.load %arg0[%arg5, %arg3, %arg4]

// CHECK-102: affine.for %arg3 = 0 to 20
// CHECK-102:   affine.for %arg4 = 0 to 10
// CHECK-102:     affine.for %arg5 = 0 to 30
// CHECK-102:        affine.load %arg0[%arg4, %arg3, %arg5]

// CHECK-012: affine.for %arg3 = 0 to 10
// CHECK-012:   affine.for %arg4 = 0 to 20
// CHECK-012:     affine.for %arg5 = 0 to 30
// CHECK-012:        affine.load %arg0[%arg3, %arg4, %arg5]

// CHECK-021: affine.for %arg3 = 0 to 10
// CHECK-021:   affine.for %arg4 = 0 to 30
// CHECK-021:     affine.for %arg5 = 0 to 20
// CHECK-021:        affine.load %arg0[%arg3, %arg5, %arg4]

// CHECK-210: affine.for %arg3 = 0 to 30
// CHECK-210:   affine.for %arg4 = 0 to 20
// CHECK-210:     affine.for %arg5 = 0 to 10
// CHECK-210:        affine.load %arg0[%arg5, %arg4, %arg3]

// CHECK-201: affine.for %arg3 = 0 to 30
// CHECK-201:   affine.for %arg4 = 0 to 10
// CHECK-201:     affine.for %arg5 = 0 to 20
// CHECK-201:        affine.load %arg0[%arg4, %arg5, %arg3]
