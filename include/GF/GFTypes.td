//===- GFTypes.td - GF dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GF_TYPES
#define GF_TYPES

include "mlir/IR/AttrTypeBase.td"
include "GF/GFDialect.td"

//===----------------------------------------------------------------------===//
// GF type definitions
//===----------------------------------------------------------------------===//

class GF_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GF_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def GF_CustomType : GF_Type<"Custom", "custom"> {
    let summary = "GF custom type";
    let description = "Custom type in gf dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // GF_TYPES
