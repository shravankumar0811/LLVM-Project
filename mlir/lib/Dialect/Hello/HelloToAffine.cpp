#include "mlir/Dialect/Hello/HelloDialect.h"
#include "mlir/Dialect/Hello/HelloOps.h"
#include "mlir/Dialect/Hello/Passes.h"

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_HELLOTOAFFINE
#include "mlir/Dialect/Hello/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hello-to-affine"

using namespace mlir;

namespace {
struct HelloToAffine : public impl::HelloToAffineBase<HelloToAffine> {

  void runOnOperation() override;
};

} // namespace

class PrintOpLowering : public mlir::OpConversionPattern<hello::PrintOp> {
  using OpConversionPattern<hello::PrintOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hello::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We don't lower "hello.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

template <typename BinaryOp, typename ArithOp>
struct BinaryOpLowering : public mlir::OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    Value result = rewriter.create<ArithOp>(loc, op->getOperands()[0],
                                            op->getOperands()[1]);

    rewriter.replaceOp(op, result);
    return success();
  }
};

using HelloAddOpLowering = BinaryOpLowering<hello::AddOp, arith::AddFOp>;
using HelloMulOpLowering = BinaryOpLowering<hello::MulOp, arith::MulFOp>;

void HelloToAffine::runOnOperation() {

  ConversionTarget target(getContext());

  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  target.addIllegalDialect<hello::HelloDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<HelloAddOpLowering, HelloMulOpLowering, PrintOpLowering>(
      &getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::hello::createHelloToAffinePass() {
  return std::make_unique<HelloToAffine>();
}