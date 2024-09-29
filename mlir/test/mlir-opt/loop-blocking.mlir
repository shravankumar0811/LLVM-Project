// RUN: mlir-opt %s -affine-loop-blocking="tile-sizes=2,2" --split-input-file | FileCheck %s --check-prefix=TILING-BY-2

// RUN: mlir-opt %s -affine-loop-blocking="tile-sizes=2,2" --split-input-file | FileCheck %s --check-prefix=DEFAULT_TILING-BY-2-2-4

module {
  func.func @matrix_copy(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %a_val = affine.load %A[%i, %j] : memref<4x4xf32>
        affine.store %a_val, %B[%i, %j] : memref<4x4xf32>
      }
    }
    return
  }
}

// TILING-BY-2: #map = affine_map<(d0) -> (d0)>
// TILING-BY-2: #map1 = affine_map<(d0) -> (d0 + 2)>
// TILING-BY-2: func.func @matrix_copy
// TILING-BY-2: affine.for %arg2 = 0 to 4 step 2 {
// TILING-BY-2: affine.apply
// TILING-BY-2: affine.apply
// TILING-BY-2: affine.for %arg3 = #map(%0) to #map1(%1) {
// TILING-BY-2: affine.for %arg4 = 0 to 4 step 2 {
// TILING-BY-2: affine.apply
// TILING-BY-2: affine.apply
// TILING-BY-2: affine.for %arg5 = #map(%2) to #map1(%3) {
// TILING-BY-2: affine.load
// TILING-BY-2: affine.store

// -----

module {
  func.func @matrix_copy(%A: memref<4x4x4xf32>, %B: memref<4x4x4xf32>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        affine.for %k = 0 to 4 {
          %a_val = affine.load %A[%i, %j, %k] : memref<4x4x4xf32>
          affine.store %a_val, %B[%i, %j, %k] : memref<4x4x4xf32>
        }
      }
    }
    return
  }
}


// DEFAULT_TILING-BY-2-2-4: #map = affine_map<(d0) -> (d0)>
// DEFAULT_TILING-BY-2-2-4: #map1 = affine_map<(d0) -> (d0 + 2)>
// DEFAULT_TILING-BY-2-2-4: #map2 = affine_map<(d0) -> (d0 + 4)>
// DEFAULT_TILING-BY-2-2-4: func.func @matrix_copy
// DEFAULT_TILING-BY-2-2-4: affine.for %arg2 = 0 to 4 step 2 {
// DEFAULT_TILING-BY-2-2-4: affine.apply
// DEFAULT_TILING-BY-2-2-4: affine.apply
// DEFAULT_TILING-BY-2-2-4: affine.for %arg3 = #map(%0) to #map1(%1) {
// DEFAULT_TILING-BY-2-2-4: affine.for %arg4 = 0 to 4 step 2 {
// DEFAULT_TILING-BY-2-2-4: affine.apply
// DEFAULT_TILING-BY-2-2-4: affine.apply
// DEFAULT_TILING-BY-2-2-4: affine.for %arg5 = #map(%2) to #map1(%3) {
// DEFAULT_TILING-BY-2-2-4: affine.for %arg6 = 0 to 4 step 4 {
// DEFAULT_TILING-BY-2-2-4: affine.apply
// DEFAULT_TILING-BY-2-2-4: affine.apply
// DEFAULT_TILING-BY-2-2-4: affine.for %arg7 = #map(%4) to #map2(%5) {
// DEFAULT_TILING-BY-2-2-4: affine.load
// DEFAULT_TILING-BY-2-2-4: affine.store