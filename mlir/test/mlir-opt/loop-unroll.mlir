// RUN: mlir-opt %s -affine-loop-unfold="unroll-factor=3" --split-input-file | FileCheck %s --check-prefix=UNROLL-BY-3
// RUN: mlir-opt %s -affine-loop-unfold="unroll-factor=2" --split-input-file | FileCheck %s --check-prefix=ONE-FOR-LOOP-UNROLL-BY-2
// RUN: mlir-opt %s -affine-loop-unfold="unroll-factor=4" --split-input-file | FileCheck %s --check-prefix=THREE-FOR-LOOP-UNROLL-BY-4

module {
  func.func @matrix_copy(%A: memref<4x4xf32>, %C: memref<4x4xf32>) {
    affine.for %i = 0 to 7 {
      affine.for %j = 0 to 5 {
        %a_val = affine.load %A[%i, %j] : memref<4x4xf32>
        affine.store %a_val, %C[%i, %j] : memref<4x4xf32>
      }
    }
    return
  }
}


// UNROLL-BY-3: #map
// UNROLL-BY-3: #map1
// UNROLL-BY-3: affine.for %arg2 = 0 to 7
// UNROLL-BY-3: affine.for %arg3 = 0 to 3 step 3
// UNROLL-BY-3: affine.load
// UNROLL-BY-3: affine.store
// UNROLL-BY-3: affine.apply 
// UNROLL-BY-3: affine.load
// UNROLL-BY-3: affine.store
// UNROLL-BY-3: affine.apply 
// UNROLL-BY-3: affine.load
// UNROLL-BY-3: affine.store

// UNROLL-BY-3: affine.for %arg3 = 3 to 5
// UNROLL-BY-3: affine.load
// UNROLL-BY-3: affine.store

// -----

module {
  func.func @static_loop_unroll_by_2(%arg0 : memref<?xf32>) {
    %0 = arith.constant 7.0 : f32
    affine.for %i0 = 0 to 20 step 1 {
      memref.store %0, %arg0[%i0] : memref<?xf32>
    }
    return
  }
}

// ONE-FOR-LOOP-UNROLL-BY-2: #map
// ONE-FOR-LOOP-UNROLL-BY-2: arith.constant
// ONE-FOR-LOOP-UNROLL-BY-2: affine.for %arg1 = 0 to 20 step 2
// ONE-FOR-LOOP-UNROLL-BY-2: memref.store
// ONE-FOR-LOOP-UNROLL-BY-2: affine.apply 
// ONE-FOR-LOOP-UNROLL-BY-2: memref.store

// -----

module {
  func.func @matrix_copy(%A: memref<4x4x4xf32>, %B: memref<4x4x4xf32>) {
    affine.for %i = 0 to 7 {
      affine.for %j = 0 to 5 {
        affine.for %k = 0 to 4 {
          %a_val = affine.load %A[%i, %j, %k] : memref<4x4x4xf32>
          affine.store %a_val, %B[%i, %j, %k] : memref<4x4x4xf32>
        }
      }
    }
    return
  }
}

// THREE-FOR-LOOP-UNROLL-BY-4: #map
// THREE-FOR-LOOP-UNROLL-BY-4: #map1
// THREE-FOR-LOOP-UNROLL-BY-4: #map2
// THREE-FOR-LOOP-UNROLL-BY-4: affine.for %arg2 = 0 to 7
// THREE-FOR-LOOP-UNROLL-BY-4: affine.for %arg3 = 0 to 5
// THREE-FOR-LOOP-UNROLL-BY-4: affine.for %arg4 = 0 to 4 step 4
// THREE-FOR-LOOP-UNROLL-BY-4: affine.load
// THREE-FOR-LOOP-UNROLL-BY-4: affine.store
// THREE-FOR-LOOP-UNROLL-BY-4: affine.apply 
// THREE-FOR-LOOP-UNROLL-BY-4: affine.load
// THREE-FOR-LOOP-UNROLL-BY-4: affine.store
// THREE-FOR-LOOP-UNROLL-BY-4: affine.apply 
// THREE-FOR-LOOP-UNROLL-BY-4: affine.load
// THREE-FOR-LOOP-UNROLL-BY-4: affine.store
// THREE-FOR-LOOP-UNROLL-BY-4: affine.apply 
// THREE-FOR-LOOP-UNROLL-BY-4: affine.load
// THREE-FOR-LOOP-UNROLL-BY-4: affine.store


