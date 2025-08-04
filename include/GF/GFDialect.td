//===- GFDialect.td - GF dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GF_DIALECT
#define GF_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GF dialect definition.
//===----------------------------------------------------------------------===//

def GF_Dialect : Dialect {
    let name = "gf";
    let summary = "A gf out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::gf";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base gf operation definition.
//===----------------------------------------------------------------------===//

class GF_Op<string mnemonic, list<Trait> traits = []> :
        Op<GF_Dialect, mnemonic, traits>;

#endif // GF_DIALECT
