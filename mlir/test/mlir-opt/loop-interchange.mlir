// RUN: mlir-opt %s --affine-loop-interchange | FileCheck %s

module {
  func.func @matrix_multiply(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        %a = affine.load %A[%i, %j] : memref<10x10xf32>
        %b = affine.load %B[%i, %j] : memref<10x10xf32>
        %c = arith.addf %a, %b : f32
        affine.store %c, %C[%i, %j] : memref<10x10xf32>
      }
    }
    return
  }
}

// CHECK: affine.for %arg3 = 0 to 20
// CHECK-NEXT: affine.for %arg4 = 0 to 10 
// CHECK-NEXT:  %0 = affine.load %arg0[%arg4, %arg3]
// CHECK-NEXT:  %1 = affine.load %arg1[%arg4, %arg3]
// CHECK-NEXT:  %2 = arith.addf %0, %1
// CHECK-NEXT:  affine.store %2, %arg2[%arg4, %arg3]
