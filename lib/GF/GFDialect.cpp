//===- GFDialect.cpp - GF dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GF/GFDialect.h"
#include "GF/GFOps.h"
#include "GF/GFTypes.h"

using namespace mlir;
using namespace mlir::gf;

#include "GF/GFOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// GF dialect.
//===----------------------------------------------------------------------===//

void GFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GF/GFOps.cpp.inc"
      >();
  registerTypes();
}
