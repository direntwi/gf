//===- GFTypes.cpp - GF dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GF/GFTypes.h"

#include "GF/GFDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::gf;

#define GET_TYPEDEF_CLASSES
#include "GF/GFOpsTypes.cpp.inc"

void GFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GF/GFOpsTypes.cpp.inc"
      >();
}
