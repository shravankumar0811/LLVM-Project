// RUN: mlir-opt  %s --affine-loop-merge --split-input-file | FileCheck %s

func.func @should_fuse_raw_dep_for_locality() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %m[%i0] : memref<10xf32>
   
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %m[%i1] : memref<10xf32>
  }
  
  return
}


// CHECK:   affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:     affine.load
// CHECK-NEXT:     affine.store 


// -----


module {
  func.func @fuse_loops(%A: memref<4xf32>, %B: memref<4xf32>, %C: memref<4xf32>) {
    affine.for %i = 0 to 4 {
      %a = affine.load %A[%i] : memref<4xf32>
      %b = affine.load %B[%i] : memref<4xf32>
      %sum = arith.addf %a, %b : f32
      affine.store %sum, %C[%i] : memref<4xf32>
    }

    affine.for %i = 0 to 4 {
      %c = affine.load %C[%i] : memref<4xf32>
      %double = arith.mulf %c, %c : f32
      affine.store %double, %C[%i] : memref<4xf32>
    }
    return
  }
}

// CHECK:      affine.for 
// CHECK-NEXT:   affine.load
// CHECK-NEXT:   affine.load
// CHECK-NEXT:   arith.addf
// CHECK-NEXT:   affine.store
// CHECK-NEXT:   affine.load
// CHECK-NEXT:   arith.mulf
// CHECK-NEXT:   affine.store  

// -----