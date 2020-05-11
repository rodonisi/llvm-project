#ifndef LLHD_BASE
#define LLHD_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLHD dialect definition
//===----------------------------------------------------------------------===//

def LLHD_Dialect : Dialect {
    let name = "llhd";

    let description = [{
        A low-level hardware description dialect.
    }];

    let cppNamespace = "llhd";
}

//===----------------------------------------------------------------------===//
// LLHD type definitions
//===----------------------------------------------------------------------===//

// LLHD sig type
class LLHD_SigType<list<Type> allowedTypes>
    : ContainerType<AnyTypeOf<allowedTypes>, CPred<"$_self.isa<SigType>()">,
        "$_self.cast<SigType>().getUnderlyingType()", "LLHD sig type">;

def LLHD_IntSigType : LLHD_SigType<[AnySignlessInteger]>;

//===----------------------------------------------------------------------===//
// LLHD op definition
//===----------------------------------------------------------------------===//

// Base class for all LLHD ops.
class LLHD_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LLHD_Dialect, mnemonic, traits> {

    // For each LLHD op, the following static functions need to be defined in
    // LLHDOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(p, *this); }];
}

#endif // LLHD_BASE
