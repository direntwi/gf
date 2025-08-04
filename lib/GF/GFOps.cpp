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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define GET_OP_CLASSES
#include "GF/GFOps.cpp.inc"

using namespace mlir;
using namespace mlir::gf;


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() {    
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

//===----------------------------------------------------------------------===//
// InvOp
//===----------------------------------------------------------------------===//

LogicalResult InvOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue)
            return emitOpError("expects a constant integer input");
        if (intValue.getInt() < 0 || intValue.getInt() > 255)
            return emitOpError("input value must be in range [0, 255]");
    }
    return success();
}
