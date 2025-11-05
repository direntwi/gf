// RUN: gf-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @add_basic
  // CHECK: gf.add %{{.*}}, %{{.*}} : i8
  // CHECK: return
  func.func @add_basic(%a: i8, %b: i8) -> i8 {
    %r = gf.add %a, %b : i8
    return %r : i8
  }

  // CHECK-LABEL: func.func @mul_basic
  // CHECK: gf.mul %{{.*}}, %{{.*}} : i8
  // CHECK: return
  func.func @mul_basic(%a: i8, %b: i8) -> i8 {
    %r = gf.mul %a, %b : i8
    return %r : i8
  }

  // CHECK-LABEL: func.func @inv_basic
  // CHECK: gf.inv %{{.*}} : i8
  func.func @inv_basic(%x: i8) -> i8 {
    %r = gf.inv %x : i8
    return %r : i8
  }

  // CHECK-LABEL: func.func @sbox_basic
  // CHECK: gf.sbox %{{.*}} : i8
  func.func @sbox_basic(%x: i8) -> i8 {
    %r = gf.sbox %x : i8
    return %r : i8
  }

  // CHECK-LABEL: func.func @inv_sbox_basic
  // CHECK: gf.inv_sbox %{{.*}} : i8
  func.func @inv_sbox_basic(%x: i8) -> i8 {
    %r = gf.inv_sbox %x : i8
    return %r : i8
  }

  // CHECK-LABEL: func.func @matmul_basic
  // CHECK: gf.matmul[{{.*}}] by[{{.*}}] into %{{.*}} {[[ATTRS:.*]]} : memref<16xi8>
  func.func @matmul_basic(%A: memref<16xi8>, %B: memref<16xi8>, %C: memref<16xi8>) {
    gf.matmul
      [%A : memref<16xi8>]
      by [%B : memref<16xi8>]
      into %C
      {rowsA = 4 : i8, colsA = 4 : i8, colsB = 4 : i8}
      : memref<16xi8>
    return
  }

  // CHECK-LABEL: func.func @mixcolumns_basic
  // CHECK: gf.mix_columns %{{.*}} into %{{.*}} : memref<4xi8>, memref<4xi8>
  func.func @mixcolumns_basic(%col_mem: memref<4xi8>, %out_col: memref<4xi8>) {
    gf.mix_columns %col_mem into %out_col : memref<4xi8>, memref<4xi8>
    return
  }

  // CHECK-LABEL: func.func @inv_mixcolumns_basic
  // CHECK: gf.inv_mix_columns %{{.*}} into %{{.*}} : memref<4xi8>, memref<4xi8>
  func.func @inv_mixcolumns_basic(%col_mem: memref<4xi8>, %out_col: memref<4xi8>) {
    gf.inv_mix_columns %col_mem into %out_col : memref<4xi8>, memref<4xi8>
    return
  }

  // CHECK-LABEL: func.func @keyschedule_basic
  // CHECK: gf.key_schedule %{{.*}} into %{{.*}} : memref<16xi8>, memref<176xi8>
  func.func @keyschedule_basic(%key: memref<16xi8>, %sched: memref<176xi8>) {
    gf.key_schedule %key into %sched : memref<16xi8>, memref<176xi8>
    return
  }

  // CHECK-LABEL: func.func @shiftrows_basic
  // CHECK: gf.shift_rows %{{.*}} : memref<16xi8>
  func.func @shiftrows_basic(%state: memref<16xi8>) {
    gf.shift_rows %state : memref<16xi8>
    return
  }

  // CHECK-LABEL: func.func @inv_shiftrows_basic
  // CHECK: gf.inv_shift_rows %{{.*}} : memref<16xi8>
  func.func @inv_shiftrows_basic(%state: memref<16xi8>) {
    gf.inv_shift_rows %state : memref<16xi8>
    return
  }

  // CHECK-LABEL: func.func @addroundkey_basic
  // CHECK: gf.add_round_key %{{.*}}, %{{.*}} : memref<16xi8>, memref<16xi8>
  func.func @addroundkey_basic(%state: memref<16xi8>, %key: memref<16xi8>) {
    gf.add_round_key %state, %key : memref<16xi8>, memref<16xi8>
    return
  }

}
