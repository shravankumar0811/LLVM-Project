//===- AffineLoopBlocking.cpp - Loop tiling pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile loop nests.
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
#define GEN_PASS_DEF_AFFINELOOPBLOCKING
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-blocking"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop Blocking pass.
struct LoopBlocking
    : public affine::impl::AffineLoopBlockingBase<LoopBlocking> {

  explicit LoopBlocking(ArrayRef<int64_t> blockList) {
    this->blockList = blockList;
  }
  void runOnOperation() override;
  void blockLoops(AffineForOp currentLoop,
                  const SmallVector<unsigned, 4> &blockMap, unsigned level,
                  OpBuilder builder, IRMapping map);
};
} // namespace

/// Recursive function to count nested AffineForOp loops.
unsigned getNestedForOpCount(AffineForOp forOp, unsigned depth = 0) {
  unsigned count = 1; // Counting the current loop as 1
  for (auto &nestedOp : forOp.getBody()->getOperations()) {
    if (auto nestedForOp = dyn_cast<AffineForOp>(nestedOp)) {
      // Recursively count inner loops
      count += getNestedForOpCount(nestedForOp, depth + 1);
    }
  }
  return count;
}

/// perfoming loop tiling for any number of loops
void LoopBlocking::blockLoops(AffineForOp currentLoop,
                              const SmallVector<unsigned, 4> &blockMap,
                              unsigned level, OpBuilder builder,
                              IRMapping map) {
  if (level >= blockMap.size())
    return; // Stop if we exceed the depth of blockMap

  // Collect loop bounds and steps
  auto lowerBound = currentLoop.getLowerBoundOperands();
  auto upperBound = currentLoop.getUpperBoundOperands();
  int64_t step = currentLoop.getStepAsInt();

  // Create a new blocked (tiled) loop at the current level.
  auto newForOp = builder.create<AffineForOp>(
      currentLoop.getLoc(), lowerBound, currentLoop.getLowerBoundMap(),
      upperBound, currentLoop.getUpperBoundMap(), step * blockMap[level],
      std::nullopt, nullptr);

  // Set insertion point to the body of the newly created loop.
  builder.setInsertionPointToStart(newForOp.getBody());

  // Define affine maps for computing the new bounds.
  AffineMap lowerMap =
      AffineMap::get(1, 0, builder.getAffineDimExpr(0), builder.getContext());
  AffineMap upperMap =
      AffineMap::get(1, 0, builder.getAffineDimExpr(0) + blockMap[level],
                     builder.getContext());

  Value newOuterIV = newForOp.getInductionVar();

  // Compute bounds using affine.apply (to be avoided in future versions).
  auto lowerBoundOp =
      builder.create<AffineApplyOp>(newForOp.getLoc(), lowerMap, newOuterIV);
  auto upperBoundOp =
      builder.create<AffineApplyOp>(newForOp.getLoc(), upperMap, newOuterIV);

  // Create a new inner loop for the blocked loop using computed bounds.
  auto newBlockForOp = builder.create<AffineForOp>(
      newForOp.getLoc(), lowerBoundOp.getResult(), lowerMap,
      upperBoundOp.getResult(), upperMap, 1);

  builder.setInsertionPointToStart(newBlockForOp.getBody());

  // Map the old induction variable to the new one in the blocked loop.
  map.map(currentLoop.getInductionVar(), newBlockForOp.getInductionVar());

  // Check for inner loops and apply tiling recursively
  for (auto &nestedOp : currentLoop.getBody()->getOperations()) {
    if (auto innerForOp = dyn_cast<AffineForOp>(nestedOp)) {
      blockLoops(innerForOp, blockMap, level + 1, builder,
                 map); // Recurse for next loop
    } else {
      // Map induction variables and clone operations if no more inner loops
      if (!isa<AffineYieldOp>(nestedOp)) {
        builder.clone(nestedOp, map);
      }
    }
  }
}

/// The main function of the LoopBlocking pass.
/// This function walks over the entire operation and identifies loops,
/// counts the number of nested loops, and applies tiling based on the tile
/// sizes.
void LoopBlocking::runOnOperation() {
  SmallVector<unsigned, 4> tilesMap(blockList.begin(), blockList.end());

  unsigned nestedLoopCount;

  // walking function to get the count of number of nested for loops
  getOperation().walk(
      [&](AffineForOp forOp) { nestedLoopCount = getNestedForOpCount(forOp); });

  // By default tiles size is 4, if for number of loops they wont give tiles
  // size then by default we will add it.
  if (tilesMap.size() != nestedLoopCount) {
    for (unsigned i = 0; i < (int)nestedLoopCount - tilesMap.size(); i++) {
      tilesMap.push_back(4);
    }
  }

  // taking only outermost for forOp and perform loop tiling.
  auto &entryBlock = getOperation().front();
  if (auto forOp = dyn_cast<AffineForOp>(entryBlock.front())) {
    OpBuilder builder(forOp);
    IRMapping map;

    // Call blockLoops to perform loop tiling.
    blockLoops(forOp, tilesMap, 0, builder, map);

    // Erase the original loop
    forOp.erase();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopBlockingPass(ArrayRef<int64_t> blockList) {
  return std::make_unique<LoopBlocking>(blockList);
}
