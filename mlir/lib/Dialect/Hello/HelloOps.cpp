#include "mlir/Dialect/Hello/HelloOps.h"
#include "mlir/Dialect/Hello/HelloDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Hello/HelloOps.cpp.inc"