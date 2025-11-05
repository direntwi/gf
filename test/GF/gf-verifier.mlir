// RUN: gf-opt %s --split-input-file --verify-diagnostics

func.func @addroundkey_invalid_state(%state: memref<8xi8>, %key: memref<16xi8>) {
    // expected-error@+1 {{state must be memref<16xi8>}}
    gf.add_round_key %state, %key : memref<8xi8>, memref<16xi8>
    return
}

// -----

func.func @addroundkey_invalid_key(%state: memref<16xi8>, %key: memref<8xi8>) {
    // expected-error@+1 {{round key must be memref<16xi8>}}
    gf.add_round_key %state, %key : memref<16xi8>, memref<8xi8>
    return
}

// -----

func.func @invalid_non_memref(%key: i8, %sched: i8) {
    // expected-error@+1 {{invalid kind of type specified: expected builtin.memref, but found 'i8'}}
    gf.key_schedule %key into %sched : i8, i8
    return
}

// -----

func.func @invalid_key_shape(%key: memref<8xi8>, %sched: memref<176xi8>) {
    // expected-error@+1 {{key must be memref<16xi8>}}
    gf.key_schedule %key into %sched : memref<8xi8>, memref<176xi8>
    return
}

// -----

func.func @invalid_sched_shape(%key: memref<16xi8>, %sched: memref<128xi8>) {
    // expected-error@+1 {{schedule must be memref<176xi8>}}
    gf.key_schedule %key into %sched : memref<16xi8>, memref<128xi8>
    return
}

// -----

func.func @invalid_element_type(%key: memref<16xi32>, %sched: memref<176xi32>) {
    // expected-error@+1 {{op operand #0 must be memref of 8-bit signless integer values, but got 'memref<16xi32>'}}
    gf.key_schedule %key into %sched : memref<16xi32>, memref<176xi32>
    return
}

// -----

func.func @invalid_alias(%key: memref<176xi8>) {
    // expected-error@+1 {{key must be memref<16xi8>}}
    gf.key_schedule %key into %key : memref<176xi8>, memref<176xi8>
    return
}

// -----

func.func @invalid_lhs_memref_size(%A: memref<8xi8>, %B: memref<16xi8>, %C: memref<16xi8>) {
    // expected-error@+1 {{lhs memref too small for rowsA × colsA}}
    gf.matmul [%A : memref<8xi8>] by [%B : memref<16xi8>] into %C
        {rowsA = 4 : i8, colsA = 4 : i8, colsB = 4 : i8}
        : memref<16xi8>
    return
}

// -----

func.func @invalid_rhs_memref_size(%A: memref<16xi8>, %B: memref<8xi8>, %C: memref<16xi8>) {
    // expected-error@+1 {{rhs memref too small for colsA × colsB}}
    gf.matmul [%A : memref<16xi8>] by [%B : memref<8xi8>] into %C
        {rowsA = 4 : i8, colsA = 4 : i8, colsB = 4 : i8}
        : memref<16xi8>
    return
}

// -----

func.func @invalid_lhs_scalar_count() {
    %a0 = arith.constant 1 : i8
    %a1 = arith.constant 2 : i8
    %b0 = arith.constant 3 : i8
    %b1 = arith.constant 4 : i8
    %out = memref.alloca() : memref<16xi8>
    // expected-error@+1 {{lhs operand size does not match rowsA × colsA}}
    gf.matmul [%a0, %a1 : i8, i8] by [%b0, %b1: i8, i8] into %out
        {rowsA = 2 : i8, colsA = 2 : i8, colsB = 2 : i8}
        : memref<16xi8>
    return
    }

// -----

func.func @invalid_rhs_scalar_count() {
    %lhs0 = arith.constant 1 : i8
    %lhs1 = arith.constant 2 : i8
    %lhs2 = arith.constant 3 : i8
    %lhs3 = arith.constant 4 : i8
    %rhs0 = arith.constant 5 : i8
    %out = memref.alloca() : memref<16xi8>
    // expected-error@+1 {{rhs operand size does not match colsA × colsB}}
    gf.matmul [%lhs0, %lhs1, %lhs2, %lhs3 : i8, i8, i8, i8]
                by [%rhs0: i8]
                into %out
                {rowsA = 2 : i8, colsA = 2 : i8, colsB = 2 : i8}
                : memref<16xi8>
    return
}

// -----

func.func @invalid_output_size(%A: memref<16xi8>, %B: memref<16xi8>, %C: memref<8xi8>) {
    // expected-error@+1 {{output memref too small for result matrix}}
    gf.matmul [%A : memref<16xi8>] by [%B : memref<16xi8>] into %C
        {rowsA = 4 : i8, colsA = 4 : i8, colsB = 4 : i8}
        : memref<8xi8>
    return
}

// -----

func.func @invalid_negative_dims(%A: memref<16xi8>, %B: memref<16xi8>, %C: memref<16xi8>) {
    // expected-error@+1 {{lhs memref too small for rowsA × colsA}}
    gf.matmul [%A : memref<16xi8>] by [%B : memref<16xi8>] into %C
        {rowsA = -4 : i8, colsA = 4 : i8, colsB = 4 : i8}
        : memref<16xi8>
    return
}

// -----

func.func @invalid_col_type(%col: memref<8xi8>, %out: memref<4xi8>) {
    // expected-error@+1 {{col operand must be memref<4xi8>}}
    gf.mix_columns %col into %out : memref<8xi8>, memref<4xi8>
    return
}

// -----

func.func @invalid_out_type(%col: memref<4xi8>, %out: memref<8xi8>) {
    // expected-error@+1 {{out operand must be memref<4xi8>}}
    gf.mix_columns %col into %out : memref<4xi8>, memref<8xi8>
    return
}

// -----

func.func @invalid_col_element_type(%col: memref<4xi32>, %out: memref<4xi8>) {
    // expected-error@+1 {{operand #0 must be memref of 8-bit signless integer values, but got 'memref<4xi32>'}}
    gf.mix_columns %col into %out : memref<4xi32>, memref<4xi8>
    return
}

// -----

func.func @invalid_out_element_type(%col: memref<4xi8>, %out: memref<4xi32>) {
    // expected-error@+1 {{operand #1 must be memref of 8-bit signless integer values, but got 'memref<4xi32>'}}
    gf.mix_columns %col into %out : memref<4xi8>, memref<4xi32>
    return
}

// -----

func.func @invalid_shape_small(%state: memref<8xi8>) {
    // expected-error@+1 {{state must be memref<16xi8>}}
    gf.shift_rows %state : memref<8xi8>
    return
}

// -----

func.func @invalid_shape_2d(%state: memref<4x4xi8>) {
    // expected-error@+1 {{state must be memref<16xi8>}}
    gf.shift_rows %state : memref<4x4xi8>
    return
}
