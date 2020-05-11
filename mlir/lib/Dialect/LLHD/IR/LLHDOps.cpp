#include "mlir/Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "mlir/Dialect/LLHD/LLHDOps.cpp.inc"
} // namespace llhd
} // namespace mlir
