// RUN: mlir-opt %s -split-input-file -verify-diagnostics -mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: @check_sig
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI64:.*]]: i64
func @check_sig(%cI1 : i1, %cI64 : i64)  {
    // CHECK-NEXT: %{{.*}} = llhd.sig "sigI1" %[[CI1]] : i1
    %sigI1 = llhd.sig "sigI1" %cI1 : i1
    // CHECK-NEXT: %{{.*}} = llhd.sig "sigI64" %[[CI64]] : i64
    %sigI64 = llhd.sig "sigI64" %cI64 : i64

    return
}

// CHECK-LABEL: @check_prb
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
func @check_prb(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>) {
    // CHECK: %{{.*}} = llhd.prb %[[SI1]] : !llhd.sig<i1>
    %0 = llhd.prb %sigI1 : !llhd.sig<i1>
    // CHECK-NEXT: %{{.*}} = llhd.prb %[[SI64]] : !llhd.sig<i64>
    %1 = llhd.prb %sigI64 : !llhd.sig<i64>
    return
}

// CHECK-LABEL: @check_drv
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI64:.*]]: i64
func @check_drv(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>, %cI1 : i1, %cI64 : i64) {
    // CHECK-NEXT: llhd.drv %[[SI1]], %[[CI1]] : !llhd.sig<i1>
    llhd.drv %sigI1, %cI1 : !llhd.sig<i1>
    // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]] : !llhd.sig<i64>
    llhd.drv %sigI64, %cI64 : !llhd.sig<i64>

    return
}

// -----

func @check_illegal_sig(%cI1 : i1) {
    // expected-error @+1 {{failed to verify that type of 'init' and underlying type of 'signal' have to match.}}
    %sig1 = "llhd.sig"(%cI1) {name="illegal"}  : (i1) -> !llhd.sig<i32>

    return
}

// -----

func @check_illegal_prb (%sig : !llhd.sig<i1>) {
    // expected-error @+1 {{failed to verify that type of 'result' and underlying type of 'signal' have to match.}}
    %prb = "llhd.prb"(%sig) {} : (!llhd.sig<i1>) -> i32

    return
}

// -----

func @check_illegal_drv (%sig : !llhd.sig<i1>, %c : i32) {
    // expected-error @+1 {{failed to verify that type of 'value' and underlying type of 'signal' have to match.}}
    "llhd.drv"(%sig, %c) {} : (!llhd.sig<i1>, i32) -> ()

    return
}
