// RUN: mlir-opt %s -affine-loop-invariant-hoist -split-input-file | FileCheck %s

func.func @nested_loops_both_having_invariant_code() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.store

  return
}

// -----

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_affine_apply
func.func @store_affine_apply() -> memref<10xf32> {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
      %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
      affine.store %cf7, %m[%t0] : memref<10xf32>
  }
  return %m : memref<10xf32>
// CHECK:       %[[cst:.*]] = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:  %[[VAR_0:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:      affine.apply
// CHECK-NEXT:      affine.store %[[cst]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[VAR_0]]  : memref<10xf32>
}

// -----

func.func @nested_loops_code_invariant_to_both() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32

  return
}

// -----

// CHECK-LABEL: func @nested_loops_inner_loops_invariant_to_outermost_loop
func.func @nested_loops_inner_loops_invariant_to_outermost_loop(%m : memref<10xindex>) {
  affine.for %arg0 = 0 to 20 {
    affine.for %arg1 = 0 to 30 {
      %v0 = affine.for %arg2 = 0 to 10 iter_args (%prevAccum = %arg1) -> index {
        %v1 = affine.load %m[%arg2] : memref<10xindex>
        %newAccum = arith.addi %prevAccum, %v1 : index
        affine.yield %newAccum : index
      }
    }
  }

  // CHECK:      affine.for %{{.*}} = 0 to 30 {
  // CHECK-NEXT:   %{{.*}}  = affine.for %{{.*}}  = 0 to 10 iter_args(%{{.*}} = %{{.*}}) -> (index) {
  // CHECK-NEXT:     %{{.*}}  = affine.load %{{.*}}[%{{.*}}  : memref<10xindex>
  // CHECK-NEXT:     %{{.*}}  = arith.addi %{{.*}}, %{{.*}} : index
  // CHECK-NEXT:     affine.yield %{{.*}} : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 20 {
  // CHECK-NEXT: }

  return
}

// -----

func.func @single_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<11xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m1[%arg0] : memref<10xf32>
    %v1 = affine.load %m2[%arg0] : memref<11xf32>
    %v2 = arith.addf %v0, %v1 : f32
    affine.store %v2, %m1[%arg0] : memref<10xf32>
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: memref.alloc() : memref<11xf32>
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.load %{{.*}} : memref<10xf32>
  // CHECK-NEXT: affine.load %{{.*}} : memref<11xf32>
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store %{{.*}} : memref<10xf32>

  return
}

// -----

func.func @invariant_code_inside_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
    affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %t0) {
        %cf9 = arith.addf %cf8, %cf8 : f32
        affine.store %cf9, %m[%arg0] : memref<10xf32>

    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.apply #map{{[0-9]*}}(%arg0)
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }


  return
}

// -----

func.func @dependent_stores() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.mulf %cf7, %cf7 : f32
      affine.store %v1, %m[%arg1] : memref<10xf32>
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %[[mul]]
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store

  return
}

// -----

func.func @independent_stores() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.mulf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg0] : memref<10xf32>
      affine.store %v1, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: %[[add:.*]] = arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: affine.store %[[mul]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:     affine.store %[[add]]
  // CHECK-NEXT:    }

  return
}

// -----

func.func @load_dependent_store() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.addf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst_0]] : f32
  // CHECK-NEXT: arith.addf %[[cst]], %[[cst]] : f32
  // CHECK-NEXT: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT: }

  return
}

// -----
