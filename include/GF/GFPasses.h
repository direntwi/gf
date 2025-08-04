//===- GFPasses.h - GF passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GF_GFPASSES_H
#define GF_GFPASSES_H

#include "GF/GFDialect.h"
#include "GF/GFOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace gf {
#define GEN_PASS_DECL
#include "GF/GFPasses.h.inc"

std::unique_ptr<Pass> createConvertGFToArithPass();

#define GEN_PASS_REGISTRATION
#include "GF/GFPasses.h.inc"
} // namespace gf
} // namespace mlir

#endif
