#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Hello//HelloOps.h"
#include "mlir/Dialect/Hello/HelloDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Hello dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Hello//HelloOpsDialect.cpp.inc"

void hello::HelloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Hello/HelloOps.cpp.inc"
      >();
}

void hello::ConstantOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  hello::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::Operation *
hello::HelloDialect::materializeConstant(mlir::OpBuilder &builder,
                                         mlir::Attribute value, mlir::Type type,
                                         mlir::Location loc) {
  return builder.create<hello::ConstantOp>(
      loc, type, value.cast<mlir::DenseElementsAttr>());
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void hello::AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult hello::AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void hello::AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void hello::MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult hello::MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void hello::MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }