// RUN: mlir-opt %s -affine-loop-unroll-fuse="unroll-jam-factor=2" --split-input-file | FileCheck %s --check-prefix=UNROLL-BY-2
// RUN: mlir-opt %s -affine-loop-unroll-fuse="unroll-jam-factor=4" --split-input-file | FileCheck %s --check-prefix=UNROLL-AND-JAM-BY-4

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

// UNROLL-BY-2: #map 
// UNROLL-BY-2: affine.for
// UNROLL-BY-2: affine.for
// UNROLL-BY-2: affine.load
// UNROLL-BY-2: affine.store
// UNROLL-BY-2: affine.apply 
// UNROLL-BY-2: affine.load
// UNROLL-BY-2: affine.store

// UNROLL-BY-2: affine.for
// UNROLL-BY-2: affine.for
// UNROLL-BY-2: affine.load
// UNROLL-BY-2: affine.store

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

// UNROLL-AND-JAM-BY-4: #map = affine_map<(d0) -> (d0 + 1)>
// UNROLL-AND-JAM-BY-4: #map1 = affine_map<(d0) -> (d0 + 2)>
// UNROLL-AND-JAM-BY-4: #map2 = affine_map<(d0) -> (d0 + 3)>
// UNROLL-AND-JAM-BY-4: affine.for
// UNROLL-AND-JAM-BY-4: affine.for
// UNROLL-AND-JAM-BY-4: affine.for
// UNROLL-AND-JAM-BY-4: affine.load
// UNROLL-AND-JAM-BY-4: affine.store
// UNROLL-AND-JAM-BY-4: affine.apply
// UNROLL-AND-JAM-BY-4: affine.load
// UNROLL-AND-JAM-BY-4: affine.store
// UNROLL-AND-JAM-BY-4: affine.apply
// UNROLL-AND-JAM-BY-4: affine.load
// UNROLL-AND-JAM-BY-4: affine.store
// UNROLL-AND-JAM-BY-4: affine.apply
// UNROLL-AND-JAM-BY-4: affine.load
// UNROLL-AND-JAM-BY-4: affine.store

// UNROLL-AND-JAM-BY-4: affine.for
// UNROLL-AND-JAM-BY-4: affine.for
// UNROLL-AND-JAM-BY-4: affine.for
// UNROLL-AND-JAM-BY-4: affine.load
// UNROLL-AND-JAM-BY-4: affine.store