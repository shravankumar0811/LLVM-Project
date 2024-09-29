//===- AffineLoopUnfold.cpp - Code to transform a loop by duplicating its body
// multiple times to reduce the number of iterations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unroll.
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
#include <cstdint>
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPUNFOLD
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-unfold"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop Unroll pass.
struct LoopUnfold : public affine::impl::AffineLoopUnfoldBase<LoopUnfold> {

  explicit LoopUnfold(int unrollFactor) { this->unrollFactor = unrollFactor; }
  void runOnOperation() override;
  void unfoldLoop(AffineForOp forOp, int unrollFactor);
};
} // namespace

static int64_t getTripCount(AffineForOp forOp) {
  auto lb = forOp.getConstantLowerBound();
  auto ub = forOp.getConstantUpperBound();
  auto step = forOp.getStepAsInt();
  return (ub - lb) / step;
}

void LoopUnfold::unfoldLoop(AffineForOp forOp, int unrollFactor) {
  OpBuilder builder(forOp);
  builder.setInsertionPoint(forOp);

  // Get loop properties
  auto ub = forOp.getConstantUpperBound();
  int64_t step = forOp.getStepAsInt();
  int64_t tripCount = getTripCount(forOp);

  // Ensure the unroll factor is valid
  if (unrollFactor <= 1 || tripCount < unrollFactor) {
    llvm::errs() << "Unroll factor is larger than the trip count or invalid.\n";
    return;
  }

  int64_t unrolledTripCount = tripCount / unrollFactor;
  int64_t remainderTripCount = tripCount % unrollFactor;
  int64_t unrolledUpperBound = ub - remainderTripCount * step;

  AffineForOp unrolledForOp;

  if (unrolledTripCount <= 0)
    return;

  // Create the unrolled loop with a step size multiplied by the unroll factor.
  unrolledForOp = builder.create<AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
      step * unrollFactor, std::nullopt, nullptr);
  unrolledForOp.setConstantUpperBound(unrolledUpperBound);

  builder.setInsertionPointToStart(unrolledForOp.getBody());

  IRMapping ivmap;
  ivmap.map(forOp.getInductionVar(), unrolledForOp.getInductionVar());
  // Clone the operations inside the loop body
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!isa<AffineYieldOp>(op)) {
      builder.clone(op, ivmap);
    }
  }

  // Get the terminator of the new loop body
  auto *innerNewForOpYieldOp = unrolledForOp.getBody()->getTerminator();

  for (int j = 1; j < unrollFactor; ++j) {
    builder.setInsertionPoint(innerNewForOpYieldOp);

    // Adjust the index based on unroll factor
    auto bumpMap = AffineMap::get(
        1, 0, builder.getAffineDimExpr(0) + j * forOp.getStepAsInt());
    auto adjustedIdx = builder.create<AffineApplyOp>(
        unrolledForOp.getLoc(), bumpMap, unrolledForOp.getInductionVar());

    // Map the outer induction variable to adjustedIdx for the unrolled
    // iteration
    ivmap.map(forOp.getInductionVar(), adjustedIdx);

    // Clone the operations from the original loop
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!isa<AffineYieldOp>(op)) {
        builder.clone(op, ivmap);
      }
    }
  }

  AffineForOp remainderForOp;
  builder.setInsertionPoint(forOp);
  // Handle the remainder loop
  if (remainderTripCount > 0) {
    remainderForOp = builder.create<AffineForOp>(
        forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), step,
        std::nullopt, nullptr);
    // Set the insertion point inside the remainder loop
    remainderForOp.setConstantLowerBound(ub - remainderTripCount * step);
    remainderForOp.setConstantUpperBound(ub);
    builder.setInsertionPointToStart(remainderForOp.getBody());

    IRMapping reminderForOpmap;
    reminderForOpmap.map(forOp.getInductionVar(),
                         remainderForOp.getInductionVar());
    // Handle the remainder iterations (no unrolling)
    for (auto &op : forOp.getBody()->without_terminator()) {
      builder.clone(op, reminderForOpmap);
    }
  }

  // Erase the original loop
  forOp.erase();
}

static bool isInnermostAffineForOp(AffineForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<AffineForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

void LoopUnfold::runOnOperation() {
  if (unrollFactor <= 1)
    return;

  // Collect all innermost loops
  SmallVector<AffineForOp, 4> loops;
  getOperation().walk([&](Operation *op) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      if (isInnermostAffineForOp(forOp)) {
        loops.push_back(forOp);
      }
    }
  });

  // Unfold each collected loop
  for (AffineForOp loop : loops) {
    unfoldLoop(loop, unrollFactor);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopUnfoldPass(int unrollFactor) {
  return std::make_unique<LoopUnfold>(unrollFactor);
}
