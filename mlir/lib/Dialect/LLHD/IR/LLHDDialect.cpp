#include "mlir/Dialect/LLHD/LLHDDialect.h"
#include "mlir/Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::llhd;

//===----------------------------------------------------------------------===//
// LLHD Dialect
//===----------------------------------------------------------------------===//

mlir::llhd::LLHDDialect::LLHDDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<SigType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLHD/LLHDOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type parsing
//===----------------------------------------------------------------------===//

/// Parse a signal type.
/// Syntax: sig ::= !llhd.sig<type>
static Type parseSigType(DialectAsmParser &parser) {
  Type underlyingType;
  if (parser.parseLess())
    return Type();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(underlyingType)) {
    parser.emitError(loc, "No signal type found. Signal needs an underlying "
                          "type.");
    return nullptr;
  }
  if (parser.parseGreater())
    return Type();
  return SigType::get(underlyingType);
}

Type LLHDDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef typeKeyword;
  // parse the type keyword first
  if (parser.parseKeyword(&typeKeyword))
    return Type();
  if (typeKeyword == SigType::getKeyword()) {
    return parseSigType(parser);
  }
  return Type();
}

//===----------------------------------------------------------------------===//
// Type printing
//===----------------------------------------------------------------------===//

/// Print a signal type with custom syntax:
/// type ::= !sig.type<underlying-type>
static void printSigType(SigType sig, DialectAsmPrinter &printer) {
  printer << sig.getKeyword() << "<";
  printer.printType(sig.getUnderlyingType());
  printer << ">";
}

void LLHDDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case LLHDTypes::Sig: {
    SigType sig = type.dyn_cast<SigType>();
    printSigType(sig, printer);
    break;
  }

  default:
    break;
  }
}

namespace mlir {
namespace llhd {
namespace detail {

//===----------------------------------------------------------------------===//
// Type storage
//===----------------------------------------------------------------------===//

// Sig Type Storage

/// Storage struct implementation for LLHD's sig type. The sig type only
/// contains one underlying llhd type.
struct SigTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;
  /// construcor for sig type's storage.
  /// Takes the underlying type as the only argument
  SigTypeStorage(mlir::Type underlyingType) : underlyingType(underlyingType) {}

  /// compare sig type instances on the underlying type
  bool operator==(const KeyTy &key) const { return key == getUnderlyingType(); }

  /// return the KeyTy for sig type
  static KeyTy getKey(mlir::Type underlyingType) {
    return KeyTy(underlyingType);
  }

  /// construction method for creating a new instance of the sig type
  /// storage
  static SigTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<SigTypeStorage>()) SigTypeStorage(key);
  }

  /// get the underlying type
  mlir::Type getUnderlyingType() const { return underlyingType; }

private:
  mlir::Type underlyingType;
};

} // namespace detail
} // namespace llhd
} // namespace mlir

//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//

// Sig Type

SigType SigType::get(mlir::Type underlyingType) {
  return Base::get(underlyingType.getContext(), LLHDTypes::Sig, underlyingType);
}

mlir::Type SigType::getUnderlyingType() {
  return getImpl()->getUnderlyingType();
}
