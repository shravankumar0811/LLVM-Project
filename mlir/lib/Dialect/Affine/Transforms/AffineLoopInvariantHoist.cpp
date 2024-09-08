//===- AffineLoopInvariantCodeMotion.cpp - Code to perform loop fusion-----===//
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
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPINVARIANTHOIST
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-invariant-hoist"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop invariant code motion (LICM) pass.
/// TODO: The pass is missing zero-trip tests.
/// TODO: This code should be removed once the new LICM pass can handle its
///       uses.
struct LoopInvariantHoist
    : public affine::impl::AffineLoopInvariantHoistBase<LoopInvariantHoist> {
  void runOnOperation() override;
  void hoistFromAffineForOp(AffineForOp forOp);
};
} // namespace

static bool isOperationLoopInvariant(Operation &op, Value indVar,
                                     ValueRange iterArgs,
                                     SmallPtrSetImpl<Operation *> &opsWithUsers,
                                     SmallPtrSetImpl<Operation *> &opsToHoist);

// Returns true if the individual op is loop invariant.
static bool isOperationLoopInvariant(Operation &op, Value indVar,
                                     ValueRange iterArgs,
                                     SmallPtrSetImpl<Operation *> &opsWithUsers,
                                     SmallPtrSetImpl<Operation *> &opsToHoist) {
  LLVM_DEBUG(llvm::dbgs() << "iterating on op: " << op;);

  // Check operands.
  for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
    auto *operandSrc = op.getOperand(i).getDefiningOp();

    LLVM_DEBUG(
        op.getOperand(i).print(llvm::dbgs() << "Iterating on operand\n"));

    // If the loop IV is the operand, this op isn't loop invariant.
    if (indVar == op.getOperand(i)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop IV is the operand\n");
      return false;
    }

    // If the one of the iter_args is the operand, this op isn't loop invariant.
    if (llvm::is_contained(iterArgs, op.getOperand(i))) {
      LLVM_DEBUG(llvm::dbgs() << "One of the iter_args is the operand\n");
      return false;
    }

    if (operandSrc) {
      LLVM_DEBUG(llvm::dbgs() << *operandSrc << "Iterating on operand src\n");

      // If the value was defined in the loop (outside of the if/else region),
      // and that operation itself wasn't meant to be hoisted, then mark this
      // operation loop dependent.
      if (opsWithUsers.count(operandSrc) && opsToHoist.count(operandSrc) == 0)
        return false;
    }
  }

  // If no operand was loop variant, mark this op for motion.
  opsToHoist.insert(&op);
  return true;
}

void LoopInvariantHoist::hoistFromAffineForOp(AffineForOp forOp) {
  auto *loopBody = forOp.getBody();
  auto indVar = forOp.getInductionVar();
  ValueRange iterArgs = forOp.getRegionIterArgs();

  // This is the place where hoisted instructions would reside.
  OpBuilder b(forOp.getOperation());

  SmallPtrSet<Operation *, 8> opsToHoist;
  SmallVector<Operation *, 8> opsToMove;
  SmallPtrSet<Operation *, 8> opsWithUsers;

  for (auto &op : *loopBody) {
    // Register op in the set of ops that have users. This set is used
    // to prevent hoisting ops that depend on these ops that are
    // not being hoisted.
    if (!op.use_empty())
      opsWithUsers.insert(&op);
    if (!isa<AffineYieldOp>(op)) {
      if (isOperationLoopInvariant(op, indVar, iterArgs, opsWithUsers,
                                   opsToHoist)) {
        opsToMove.push_back(&op);
      }
    }
  }

  // For all instructions that we found to be invariant, place sequentially
  // right before the for loop.
  for (auto *op : opsToMove) {
    op->moveBefore(forOp);
  }

  LLVM_DEBUG(forOp->print(llvm::dbgs() << "Modified loop\n"));
}

void LoopInvariantHoist::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order.  This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation().walk([&](AffineForOp op) {
    LLVM_DEBUG(op->print(llvm::dbgs() << "\nOriginal loop\n"));
    hoistFromAffineForOp(op);
  });
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopInvariantHoistPass() {
  return std::make_unique<LoopInvariantHoist>();
}
