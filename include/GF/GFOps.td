//===- GFOps.td - GF dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GF_OPS
#define GF_OPS

include "GF/GFTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GF_FooOp : GF_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `gf.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = gf.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def GF_AddOp : GF_Op<"add", [Pure]> {
    let summary = "Addition in GF(2^8) using XOR.";
    let description = [{}];
    let hasVerifier = 1;
    let arguments = (ins I8:$lhs, I8:$rhs);
    let results = (outs I8:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def GF_MulOp : GF_Op<"mul", [Pure]> {
    let summary = "Multiplication in GF(2^8)";
    let description = [{
        Performs multiplication in GF(2^8) via log/antilog table lookups.
        The op takes two GF field elements and produces a GF field element result.
    }];
    let hasVerifier = 1;
    let arguments = (ins I8:$lhs, I8:$rhs);
    let results = (outs I8:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// InvOp
//===----------------------------------------------------------------------===//

def GF_InvOp : GF_Op<"inv", [Pure]> {
    let summary = "Multiplicative inverse in GF(2^8)";
    let description = [{
        Computes the multiplicative inverse of a non‑zero GF(2^8) element
        by performing a log lookup, subtracting from 255, then doing an antilog lookup.
    }];

    let arguments = (ins I8:$input);
    let results = (outs I8:$res);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

def GF_MatMulOp : GF_Op<"matmul", [AttrSizedOperandSegments]> {
    let summary = "Matrix multiplication in GF(2^8)";
    let description = [{
        Given A ∈ GF(2^8)^(M×K) and B ∈ GF(2^8)^(K×N), computes
        C = A · B   with C stored in the provided output MemRef.
        Inputs are flattened row-major lists.
    }];

    let arguments = (ins
        Variadic<AnyTypeOf<[I8, MemRefOf<[I8]>]>>:$lhs,
        Variadic<AnyTypeOf<[I8, MemRefOf<[I8]>]>>:$rhs,
        MemRefOf<[I8]>:$output,
        I8Attr:$rowsA,
        I8Attr:$colsA,
        I8Attr:$colsB
    );

    let assemblyFormat = [{
        `[` $lhs `:` type($lhs) `]`
        `by`
        `[` $rhs `:` type($rhs) `]`
        `into` $output attr-dict `:` type($output)
    }];

    let hasVerifier = 1;
}

#endif // GF_OPS
