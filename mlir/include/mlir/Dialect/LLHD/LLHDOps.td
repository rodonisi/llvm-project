#ifndef LLHD_OPS
#define LLHD_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLHD/LLHDBase.td"

//===----------------------------------------------------------------------===//
//=== Signal Operations
//===----------------------------------------------------------------------===//

def LLHD_SigOp : LLHD_Op<"sig",
        [TypesMatchWith<"type of 'init' and underlying type of 'signal' have to match.",
                        "init", "result",
                        "SigType::get($_self)">]> {
    let summary = "Create a signal.";

    let description = [{
       The `llhd.sig` instruction introduces a new signal in the IR. The input
       operand determines the initial value carried by the signal, while the
       result type will always be a signal carrying the type of the init operand.
       A signal defines a name that can be used to identify it.

       **Custom syntax:**

       ```
       sig-op ::= ssa-id `=` `llhd.sig` sig-name ssa-init attr-dict `:` init-type
       ```

       **Examples:**

       ```
       %init_i64 = constant 123 : i64
       %sig_i64 = llhd.sig "foo" %init_64 : i64

       %init_i1 = constant 1 : i1
       %sig_i1 = llhd.sig "bar" %init_i1 : i1
       ```
       The first `llhd.sig` instruction creates a new signal named "foo", carrying an `i64`
       type with initial value of 123, while the second one creates a new signal
       named "bar", carrying an `i1` type with initial value of 1.
    }];

    let arguments = (ins StrAttr:$name, AnySignlessInteger:$init);
    let results = (outs LLHD_IntSigType:$result);

    let assemblyFormat = "$name $init attr-dict `:` type($init)";
}

def LLHD_PrbOp : LLHD_Op<"prb",
        [NoSideEffect,
         TypesMatchWith<"type of 'result' and underlying type of 'signal' have to match.",
                        "signal", "result",
                        "$_self.cast<SigType>().getUnderlyingType()">]> {
    let summary = "Probe a signal.";

    let description = [{
       The `llhd.prb` instruction probes a signal and returns the value it
       currently carries as a new SSA operand. The result type is always
       the type carried by the signal.

        **Custom syntax:**

        ```
        prb-op ::= ssa-id `=` `llhd.prb` ssa-sig attr-dict `:` !llhd.sig<type>
        ```

        **Examples:***

        ```
        %const_i1 = constant 1 : i1
        %sig_i1 = llhd.sig %const_i1 : i1
        %prbd = llhd.prb %sig_i1 : !llhd.sig<i1>
        ```
    }];

    let arguments = (ins LLHD_IntSigType:$signal);
    let results = (outs AnySignlessInteger:$result);

    let assemblyFormat = "$signal attr-dict `:` type($signal)";
}

def LLHD_DrvOp : LLHD_Op<"drv",
        [TypesMatchWith<"type of 'value' and underlying type of 'signal' have to match.",
                        "signal", "value",
                        "$_self.cast<SigType>().getUnderlyingType()">]> {
    let summary = "Drive a value into a signal.";

    let description = [{
        The `llhd.drv` operation schedules the specified signal to change to the
        new value after a delay, which is implicitly defined as a delta step.
        This operation does not define any new SSA operands.

        **Custom Syntax:**

        ```
        drv-op ::= `llhd.drv` ssa-signal `,` ssa-const `:` !llhd.sig<const-type>
        ```

        **Examples:**

        ```
        %init = constant 1 : i1
        %sig = llhd.sig %init : i1

        llhd.drv %sig, %new : !llhd.sig<i1>
        llhd.drv %sig, %new : !llhd.sig<i1>
        ```
    }];

    let arguments = (ins LLHD_IntSigType:$signal,
                         AnySignlessInteger:$value);

    let assemblyFormat = "operands attr-dict `:` type($signal)";
}

#endif // LLHD_OPS
