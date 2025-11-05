// RUN: gf-opt %s --pass-pipeline="builtin.module(gf-switch-bar-foo)" | FileCheck %s --check-prefix=CHECK-BAR
// RUN: gf-opt %s --pass-pipeline="builtin.module(convert-gf-to-arith)" | FileCheck %s --check-prefix=CHECK-CONVERT

module {
  // CHECK-BAR-LABEL: func.func @foo()
  func.func @bar() {
    return
  }

  // CHECK-BAR-LABEL: func.func @abar()
  func.func @abar() {
    return
  }
}

module {
  // CHECK-CONVERT-LABEL: func.func @lower_arith_memref_ops
  // CHECK-CONVERT-NOT: gf.add
  // CHECK-CONVERT-NOT: gf.mul
  // CHECK-CONVERT-NOT: gf.inv
  // CHECK-CONVERT-NOT: gf.sbox
  // CHECK-CONVERT-NOT: gf.inv_sbox
  // CHECK-CONVERT: arith.xori
  // CHECK-CONVERT: memref.
  func.func @lower_arith_memref_ops() {
    %a = arith.constant 5 : i8
    %b = arith.constant 7 : i8
    %c = gf.add %a, %b : i8
    %d = gf.mul %a, %b : i8
    %e = gf.inv %a :i8
    %f = gf.sbox %b :i8
    %g = gf.inv_sbox %b :i8
    func.return
  }

  // CHECK-CONVERT-LABEL: func.func @mix_columns
  // CHECK-CONVERT-NOT: gf.mix_columns
  // CHECK-CONVERT: scf.
  func.func @mix_columns(%state: memref<16xi8>, %out: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i4  = arith.constant 4  : index
    %col_mem = memref.alloca() : memref<4xi8>
    %out_col = memref.alloca() : memref<4xi8>

    scf.for %col = %i0 to %i4 step %i1 {
      %v0 = memref.load %state[%i0] : memref<16xi8>
      memref.store %v0, %col_mem[%i0] : memref<4xi8>
      gf.mix_columns %col_mem into %out_col : memref<4xi8>, memref<4xi8>
      %o0 = memref.load %out_col[%i0] : memref<4xi8>
      memref.store %o0, %out[%i0] : memref<16xi8>
    }
    return
  }

  // CHECK-CONVERT-LABEL: func.func @inv_mix_columns
  // CHECK-CONVERT-NOT: gf.inv_mix_columns
  // CHECK-CONVERT: scf.
  func.func @inv_mix_columns(%state: memref<16xi8>, %out: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i4  = arith.constant 4  : index
    %col_mem = memref.alloca() : memref<4xi8>
    %out_col = memref.alloca() : memref<4xi8>

    scf.for %col = %i0 to %i4 step %i1 {
      %v0 = memref.load %state[%i0] : memref<16xi8>
      memref.store %v0, %col_mem[%i0] : memref<4xi8>
      gf.inv_mix_columns %col_mem into %out_col : memref<4xi8>, memref<4xi8>
      %o0 = memref.load %out_col[%i0] : memref<4xi8>
      memref.store %o0, %out[%i0] : memref<16xi8>
    }
    return
  }

  // CHECK-CONVERT-LABEL: func.func @key_schedule
  // CHECK-CONVERT-NOT: gf.key_schedule
  // CHECK-CONVERT: scf.
  func.func @key_schedule(%key: memref<16xi8>, %sched: memref<176xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i16 = arith.constant 16 : index

    scf.for %t = %i0 to %i16 step %i1 {
      gf.key_schedule %key into %sched : memref<16xi8>, memref<176xi8>
    }
    return
  }

  // CHECK-CONVERT-LABEL: func.func @shift_rows
  // CHECK-CONVERT-NOT: gf.shift_rows
  // CHECK-CONVERT: scf.
  func.func @shift_rows(%state: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i16 = arith.constant 16 : index

    scf.for %r = %i0 to %i16 step %i1 {
      gf.shift_rows %state : memref<16xi8>
    }
    return
  }

  // CHECK-CONVERT-LABEL: func.func @inv_shift_rows
  // CHECK-CONVERT-NOT: gf.inv_shift_rows
  // CHECK-CONVERT: scf.
  func.func @inv_shift_rows(%state: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i16 = arith.constant 16 : index

    scf.for %r = %i0 to %i16 step %i1 {
      gf.inv_shift_rows %state : memref<16xi8>
    }
    return
  }

  // CHECK-CONVERT-LABEL: func.func @add_round_key
  // CHECK-CONVERT-NOT: gf.add_round_key
  // CHECK-CONVERT: scf.
  func.func @add_round_key(%state: memref<16xi8>, %key: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i16 = arith.constant 16 : index

    scf.for %i = %i0 to %i16 step %i1 {
      gf.add_round_key %state, %key : memref<16xi8>, memref<16xi8>
    }
    return
  }

}
