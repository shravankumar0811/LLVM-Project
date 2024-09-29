//===- AffineLoopSwap.cpp - Code to Interchnage the order of loops
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop interchange the order of 2 affine for loops.
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
#define GEN_PASS_DEF_AFFINELOOPSWAP
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-interchange"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop Swap pass.
struct LoopSwap : public affine::impl::AffineLoopSwapBase<LoopSwap> {
  void runOnOperation() override;
  void swapLoopsUpstream(AffineForOp outerForOp, AffineForOp innerForOp);
  void swapLoops(AffineForOp outerForOp, AffineForOp innerForOp);
};
} // namespace

void LoopSwap::swapLoopsUpstream(AffineForOp outerForOp,
                                 AffineForOp innerForOp) {
  assert(&*outerForOp.getBody()->begin() == innerForOp.getOperation());
  auto &innerForOpBody = innerForOp.getBody()->getOperations();

  // Step 1 - move 'innerForOp' itself (without its body) out of 'outerForOp'
  outerForOp->getBlock()->getOperations().splice(
      Block::iterator(outerForOp),           // Move just before outerForOp
      outerForOp.getBody()->getOperations(), // Outer loop body
      outerForOp.getBody()->begin(), // Move innerForOp (first operation)
      std::prev(outerForOp.getBody()->end()) // Exclude the terminator
  );

  // Step 2 - move the contents of 'innerForOp' into 'outerForOp'
  outerForOp.getBody()->getOperations().splice(
      outerForOp.getBody()
          ->begin(), // Move to the beginning of outerForOp's body
      innerForOp.getBody()->getOperations(), // Inner loop body
      innerForOp.getBody()->begin(),         // Start of innerForOp's body
      std::prev(innerForOp.getBody()->end()) // Exclude innerForOp's terminator
  );

  // 3) Splice outerForOp into the beginning of innerForOp's body.
  innerForOpBody.splice(innerForOpBody.begin(),
                        outerForOp->getBlock()->getOperations(),
                        Block::iterator(outerForOp));
}

void LoopSwap::swapLoops(AffineForOp outerForOp, AffineForOp innerForOp) {

  OpBuilder builder(outerForOp);

  // Collect loop bounds and steps
  auto outerLowerBound = outerForOp.getLowerBoundOperands();
  auto outerUpperBound = outerForOp.getUpperBoundOperands();
  int64_t outerStep = outerForOp.getStepAsInt();

  auto innerLowerBound = innerForOp.getLowerBoundOperands();
  auto innerUpperBound = innerForOp.getUpperBoundOperands();
  int64_t innerStep = innerForOp.getStepAsInt();

  // Create new Swapd loops
  auto newOuterForOp = builder.create<AffineForOp>(
      outerForOp.getLoc(), innerLowerBound, innerForOp.getLowerBoundMap(),
      innerUpperBound, innerForOp.getUpperBoundMap(), innerStep, std::nullopt,
      nullptr);

  builder.setInsertionPointToStart(newOuterForOp.getBody());

  auto newInnerForOp = builder.create<AffineForOp>(
      innerForOp.getLoc(), outerLowerBound, outerForOp.getLowerBoundMap(),
      outerUpperBound, outerForOp.getUpperBoundMap(), outerStep, std::nullopt,
      nullptr);

  IRMapping map;
  map.map(outerForOp.getInductionVar(), newInnerForOp.getInductionVar());
  map.map(innerForOp.getInductionVar(), newOuterForOp.getInductionVar());

  builder.setInsertionPointToStart(newInnerForOp.getBody());

  for (auto &op : innerForOp.getBody()->without_terminator()) {
    if (!isa<AffineYieldOp>(op)) {
      builder.clone(op, map);
    }
  }

  // Clean up the old loops
  innerForOp.erase();
  outerForOp.erase();
}

void LoopSwap::runOnOperation() {
  // Define a map to store outer and inner loop pairs
  llvm::DenseMap<AffineForOp, AffineForOp> loopMap;

  // Walk through the operations and populate the map with outer-inner pairs
  getOperation().walk([&](AffineForOp outerForOp) {
    LLVM_DEBUG(outerForOp->print(llvm::dbgs() << "\nOriginal loop\n"));

    for (auto &nestedOp : outerForOp.getBody()->getOperations()) {
      if (auto innerForOp = dyn_cast<AffineForOp>(nestedOp)) {
        // Store the outer and inner loop pair in the map
        loopMap[outerForOp] = innerForOp;
      }
    }
  });

  // Now, iterate over the map and call the SwapLoops function for each
  // pair
  for (auto &entry : loopMap) {
    AffineForOp outerForOp = entry.first;
    AffineForOp innerForOp = entry.second;

    LLVM_DEBUG(llvm::dbgs() << "Performing loop Swap\n");

    // Call the Swap function using the map contents
    // swapLoopsUpstream(outerForOp, innerForOp);
    swapLoops(outerForOp, innerForOp);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopSwapPass() {
  return std::make_unique<LoopSwap>();
}
