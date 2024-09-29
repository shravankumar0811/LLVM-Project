//===- AffineLoopPermute.cpp - Code to interchange the order of loops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop invariant code motion.
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
#define GEN_PASS_DEF_AFFINELOOPPERMUTE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-permute"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop permute pass.
struct LoopPermute : public affine::impl::AffineLoopPermuteBase<LoopPermute> {

  explicit LoopPermute(ArrayRef<int64_t> permList) {
    this->permList = permList;
  }
  void runOnOperation() override;
  void permuteLoops(const llvm::DenseMap<int, AffineForOp> &loopMap,
                    const SmallVector<unsigned, 4> &permMap);
};
} // namespace

void LoopPermute::permuteLoops(const llvm::DenseMap<int, AffineForOp> &loopMap,
                               const SmallVector<unsigned, 4> &permMap) {

  assert(loopMap.size() == permMap.size() &&
         "Loop and permutation sizes must match!");

  AffineForOp firstLoop = loopMap.lookup(0);
  OpBuilder builder(firstLoop);

  // Initialize IRMapping to map old induction variables to new ones
  IRMapping ivMap;

  // New outermost loop
  AffineForOp firstLoopOp = loopMap.lookup(permMap[0]);
  auto outerLowerBound = firstLoopOp.getLowerBoundOperands();
  auto outerUpperBound = firstLoopOp.getUpperBoundOperands();
  int64_t outerStep = firstLoopOp.getStepAsInt();

  auto newFirstLoop = builder.create<AffineForOp>(
      firstLoop.getLoc(), outerLowerBound, firstLoopOp.getLowerBoundMap(),
      outerUpperBound, firstLoopOp.getUpperBoundMap(), outerStep, std::nullopt,
      nullptr);

  builder.setInsertionPointToStart(newFirstLoop.getBody());
  // Map the old induction variable to the new one in the blocked loop.
  ivMap.map(firstLoopOp.getInductionVar(), newFirstLoop.getInductionVar());

  // Iterate over remaining loops and permute based on permMap
  AffineForOp prevLoop = newFirstLoop;
  for (size_t i = 1; i < permMap.size(); ++i) {
    AffineForOp loopOp = loopMap.lookup(permMap[i]);
    auto lowerBound = loopOp.getLowerBoundOperands();
    auto upperBound = loopOp.getUpperBoundOperands();
    int64_t step = loopOp.getStepAsInt();

    auto newLoop = builder.create<AffineForOp>(
        loopOp.getLoc(), lowerBound, loopOp.getLowerBoundMap(), upperBound,
        loopOp.getUpperBoundMap(), step, std::nullopt, nullptr);

    // Map induction variable
    ivMap.map(loopOp.getInductionVar(), newLoop.getInductionVar());

    builder.setInsertionPointToStart(newLoop.getBody());
    prevLoop = newLoop;
  }

  // Set the innermost loop correctly based on permutation
  AffineForOp innermostLoop = loopMap.lookup(loopMap.size() - 1);

  builder.setInsertionPointToStart(prevLoop.getBody());

  // Clone the operations from the innermost loop
  for (auto &op : innermostLoop.getBody()->without_terminator()) {
    if (!isa<AffineYieldOp>(op)) {
      builder.clone(op, ivMap);
    }
  }

  // Erase original loops
  for (int i = loopMap.size() - 1; i >= 0; --i) {
    loopMap.lookup(i).erase();
  }
}

void LoopPermute::runOnOperation() {
  SmallVector<unsigned, 4> permMap(permList.begin(), permList.end());

  SmallVector<AffineForOp, 4> loops;
  getOperation()->walk([&](AffineForOp forOp) {
    AffineForOp currentLoop = forOp;

    // Collect all the loops in the nest
    while (currentLoop) {
      loops.push_back(currentLoop);
      if (currentLoop.getBody()->getOperations().size() != 1 ||
          !llvm::isa<AffineForOp>(currentLoop.getBody()->front())) {
        break;
      }
      currentLoop = llvm::cast<AffineForOp>(currentLoop.getBody()->front());
    }
  });

  llvm::DenseMap<int, AffineForOp> loopMap;
  int j = 0;
  for (int i = (int)loops.size() - 1; i >= 0; i--) {
    loopMap[j] = loops[i];
    j++;
  }

  permuteLoops(loopMap, permMap);

}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopPermutePass(ArrayRef<int64_t> permList) {
  return std::make_unique<LoopPermute>(permList);
}
