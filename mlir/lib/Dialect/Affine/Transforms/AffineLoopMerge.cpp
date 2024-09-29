//===- AffineLoopMerge.cpp - Code to merging of 2 loops
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop fusiion of 2 affine for loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPMERGE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-merge"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop Swap pass.
struct LoopMerge : public affine::impl::AffineLoopMergeBase<LoopMerge> {
  void runOnOperation() override;
  bool fusionCheck(AffineForOp forOp1, AffineForOp forOp2);
  void fusionLoops(AffineForOp forOp1, AffineForOp forOp2);
};
} // namespace

bool LoopMerge::fusionCheck(AffineForOp forOp1, AffineForOp forOp2) {
  // Check if the two loops are adjacent.
  if (forOp1->getParentOp() != forOp2->getParentOp()) {
    llvm::errs() << "The two loops are not adjacent. CANNOT fuse.\n";
    return false;
  }

  // Check if the lower bound integer are same.
  if (forOp1.getConstantLowerBound() != forOp2.getConstantLowerBound()) {
    llvm::errs() << "The lower bound values of 2 loops "
                 << "are not same. CANNOT fuse.\n";
    return false;
  }

  // Check if the upper bound integer are same.
  if (forOp1.getConstantUpperBound() != forOp2.getConstantUpperBound()) {
    llvm::errs() << "The upper bound values of 2 loops "
                 << "are not same. CANNOT fuse.\n";
    return false;
  }

  // Check if the steps are same.
  if (forOp1.getStepAsInt() != forOp2.getStepAsInt()) {
    llvm::errs() << "The step values of 2 loops "
                 << "are not same. CANNOT fuse.\n";
    return false;
  }

  llvm::errs() << "The checks have all passed. The loops can be fused.\n";
  return true;
}

void LoopMerge::fusionLoops(AffineForOp forOp1, AffineForOp forOp2) {

  OpBuilder builder(forOp1);
  IRMapping map;
  builder.setInsertionPointToStart(forOp1.getBody());
  map.map(forOp2.getInductionVar(), forOp1.getInductionVar());
  for (auto &op : forOp2.getBody()->without_terminator()) {
    if (!isa<AffineYieldOp>(op)) {
      builder.clone(op, map);
    }
  }

  forOp2->erase();
}

void LoopMerge::runOnOperation() {
  // Define a map to store outer and inner loop pairs
  llvm::SmallVector<AffineForOp, 4> forOps;

  // Walk through the operations and populate the map with outer-inner pairs
  getOperation().walk([&](Operation *op) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      // Store the outer and inner loop pair in the map
      forOps.push_back(forOp);
    }
  });

  if (forOps.size() < 2) {
    llvm::errs() << "The program contains less no of loops to fuse\n";
    return signalPassFailure();
  }

  // Check for each combinations of loops are fusable
  for (int i = 0; i < (int)forOps.size(); i++) {
    for (int j = i + 1; j < (int)forOps.size(); j++) {
      // Function to perform basic checks on the two loops
      if (fusionCheck(forOps[j], forOps[i]))
        // Function to perform fusing on the two loops
        fusionLoops(forOps[j], forOps[i]);
      break;
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopMergePass() {
  return std::make_unique<LoopMerge>();
}
