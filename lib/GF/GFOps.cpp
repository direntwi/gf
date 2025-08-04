//===- GFOps.cpp - GF dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GF/GFOps.h"
#include "GF/GFDialect.h"
#include "mlir/IR/Matchers.h"

#define GET_OP_CLASSES
#include "GF/GFOps.cpp.inc"

using namespace mlir;
using namespace mlir::gf;


LogicalResult AddOp::verify() {   
    auto checkOperand = [&](Value operand) -> LogicalResult {
        IntegerAttr value;
        if (matchPattern(operand, m_Constant(&value))) {
            int64_t val = value.getValue().getSExtValue();
            if (val < 0 || val > 255) {
                return emitOpError()
                    << "operand value " << val << " out of range [0,255]";
            }
        }
        return success();
    };
    if (failed(checkOperand(getLhs())) || failed(checkOperand(getRhs()))) {
        return failure();
    }
    return success();
}