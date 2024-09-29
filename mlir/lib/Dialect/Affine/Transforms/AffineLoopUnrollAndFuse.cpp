//===- AffineLoopUnrollAndFuse.cpp - Code to perform loop unroll and
// jam-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unroll and jam. Unroll and jam is a transformation
// that improves locality, in particular, register reuse, while also improving
// operation level parallelism. The example below shows what it does in nearly
// the general case. Loop unroll and jam currently works if the bounds of the
// loops inner to the loop being unroll-jammed do not depend on the latter.
//
// Before      After unroll and jam of i by factor 2:
//
//             for i, step = 2
// for i         S1(i);
//   S1;         S2(i);
//   S2;         S1(i+1);
//   for j       S2(i+1);
//     S3;       for j
//     S4;         S3(i, j);
//   S5;           S4(i, j);
//   S6;           S3(i+1, j)
//                 S4(i+1, j)
//               S5(i);
//               S6(i);
//               S5(i+1);
//               S6(i+1);
//
// Note: 'if/else' blocks are not jammed. So, if there are loops inside if
// op's, bodies of those loops will not be jammed.
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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPUNROLLANDFUSE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-unroll-fuse"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop Unroll and Fuse pass.
/// This pass performs loop unrolling and jamming on affine loops, which can
/// improve cache usage and instruction-level parallelism by reducing loop
/// overhead and improving data locality.
struct LoopUnrollAndFuse
    : public affine::impl::AffineLoopUnrollAndFuseBase<LoopUnrollAndFuse> {

  explicit LoopUnrollAndFuse(int unrollJamFactor) {
    this->unrollJamFactor = unrollJamFactor;
  }
  void runOnOperation() override;
  void unrollAndFuseLoop(AffineForOp forOp, int unrollJamFactor);
  void recursiveUnrollAndFuse(AffineForOp outerForOp,
                              AffineForOp unrollAndJamForOp,
                              int unrollJamFactor, OpBuilder builder,
                              IRMapping ivmap, Value outerOldForOpIv,
                              Value outerNewForOpIv);
};
} // namespace

/// Returns the trip count (number of iterations) of the given affine loop.
static int64_t getTripCount(AffineForOp forOp) {
  auto lb = forOp.getConstantLowerBound();
  auto ub = forOp.getConstantUpperBound();
  auto step = forOp.getStepAsInt();
  return (ub - lb) / step;
}

/// Retrieves the innermost loop of a nested loop structure starting from the
/// given outer loop.
AffineForOp getInnerLoop(AffineForOp outerForOp) {
  // Iterate through the operations in the outer loop's body
  for (Operation &op : outerForOp.getBody()->without_terminator()) {
    // Check if the operation is an AffineForOp (inner loop)
    if (auto innerForOp = dyn_cast<AffineForOp>(&op)) {
      return innerForOp;
    }
  }
  // Return null if no inner loop is found
  return nullptr;
}

void LoopUnrollAndFuse::unrollAndFuseLoop(AffineForOp forOp,
                                          int unrollJamFactor) {
  auto ub = forOp.getConstantUpperBound();
  int64_t step = forOp.getStepAsInt();
  int64_t tripCount = getTripCount(forOp);

  // Ensure the unrollJam factor is valid
  if (unrollJamFactor <= 1 || tripCount < unrollJamFactor) {
    llvm::errs() << "Unroll factor is larger than the trip count or invalid.\n";
    return;
  }

  OpBuilder builder(forOp);
  builder.setInsertionPoint(forOp);

  int64_t unrollAndJammedTripCount = tripCount / unrollJamFactor;
  int64_t remainderTripCount = tripCount % unrollJamFactor;
  int64_t unrolledUpperBound = ub - remainderTripCount * step;

  if (unrollAndJammedTripCount <= 0)
    return;

  IRMapping ivmap;
  // Create the unrolled and jammed loop for the outer loop
  AffineForOp unrollAndJamForOp = builder.create<AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
      step * unrollJamFactor, std::nullopt, nullptr);
  unrollAndJamForOp.setConstantUpperBound(unrolledUpperBound);

  builder.setInsertionPointToStart(unrollAndJamForOp.getBody());
  ivmap.map(forOp.getInductionVar(), unrollAndJamForOp.getInductionVar());
  llvm::SmallVector<AffineForOp, 4> forOps;
  forOps.push_back(forOp);
  // Recursively unroll and fuse the inner loops
  recursiveUnrollAndFuse(forOp, unrollAndJamForOp, unrollJamFactor, builder,
                         ivmap, forOp.getInductionVar(),
                         unrollAndJamForOp.getInductionVar());

  builder.setInsertionPoint(forOp);
  // Handle the remainder loop
  if (remainderTripCount > 0) {
    AffineForOp remainderForOp = builder.create<AffineForOp>(
        forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), step,
        std::nullopt, nullptr);

    remainderForOp.setConstantLowerBound(ub - remainderTripCount * step);
    remainderForOp.setConstantUpperBound(ub);
    builder.setInsertionPointToStart(remainderForOp.getBody());

    IRMapping map;
    map.map(forOp.getInductionVar(), remainderForOp.getInductionVar());

    // Clone the body of the original loop for remainder iterations
    for (auto &op : forOp.getBody()->without_terminator()) {
      builder.clone(op, map);
    }
  }

  // Erase the original loop
  forOp.erase();
}

void LoopUnrollAndFuse::recursiveUnrollAndFuse(
    AffineForOp outerForOp, AffineForOp unrollAndJamForOp, int unrollJamFactor,
    OpBuilder builder, IRMapping ivmap, Value outerOldForOpIv,
    Value outerNewForOpIv) {
  // Get the next inner loop
  AffineForOp innerOldForOp = getInnerLoop(outerForOp);

  if (!innerOldForOp) {

    // No inner loop, we clone the body and map the induction variables
    builder.setInsertionPointToStart(unrollAndJamForOp.getBody());

    // Clone the operations inside the loop body
    for (auto &op : outerForOp.getBody()->without_terminator()) {
      if (!isa<AffineYieldOp>(op)) {
        builder.clone(op, ivmap);
      }
    }

    // Get the terminator of the new loop body
    auto *innerNewForOpYieldOp = unrollAndJamForOp.getBody()->getTerminator();

    // Unroll and fuse the body for the specified unrollJamFactor
    for (int i = 1; i < unrollJamFactor; ++i) {
      builder.setInsertionPoint(innerNewForOpYieldOp);

      // Adjust the index based on unroll factor
      auto bumpMap = AffineMap::get(
          1, 0, builder.getAffineDimExpr(0) + i * outerForOp.getStepAsInt());
      auto adjustedIdx = builder.create<AffineApplyOp>(
          unrollAndJamForOp.getLoc(), bumpMap, outerNewForOpIv);

      // Map the outer induction variable to adjustedIdx for the unrolled
      // iteration
      ivmap.map(outerOldForOpIv, adjustedIdx);

      for (auto &op : outerForOp.getBody()->without_terminator()) {
        if (!isa<AffineYieldOp>(op)) {
          builder.clone(op, ivmap);
        }
      }
    }

    return; // Exit recursion as we've reached the innermost loop
  }

  // If there's an inner loop, unroll and fuse it as well
  AffineForOp innerNewForOp = builder.create<AffineForOp>(
      outerForOp.getLoc(), innerOldForOp.getLowerBoundOperands(),
      innerOldForOp.getLowerBoundMap(), innerOldForOp.getUpperBoundOperands(),
      innerOldForOp.getUpperBoundMap(), innerOldForOp.getStepAsInt(),
      std::nullopt, nullptr);

  // Set insertion point to the start of the new loop
  builder.setInsertionPointToStart(innerNewForOp.getBody());

  ivmap.map(innerOldForOp.getInductionVar(), innerNewForOp.getInductionVar());
  // Recursively process the inner loop
  recursiveUnrollAndFuse(innerOldForOp, innerNewForOp, unrollJamFactor, builder,
                         std::move(ivmap), outerOldForOpIv, outerNewForOpIv);
}

/// Main function to run the unroll and fuse pass on the operation.
void LoopUnrollAndFuse::runOnOperation() {
  if (unrollJamFactor <= 1)
    return;

  auto &entryBlock = getOperation().front();
  if (auto forOp = dyn_cast<AffineForOp>(entryBlock.front())) {
    unrollAndFuseLoop(forOp, unrollJamFactor);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopUnrollAndFusePass(int unrollJamFactor) {
  return std::make_unique<LoopUnrollAndFuse>(unrollJamFactor);
}
