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

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsVals = getLhs();
  auto rhsVals = getRhs();
  auto outputType = mlir::dyn_cast<MemRefType>(getOutput().getType());

  int64_t M = getRowsA();
  int64_t K = getColsA();
  int64_t N = getColsB();

  // Check dimensions are positive
  if (M <= 0 || K <= 0 || N <= 0)
    return emitOpError("all matrix dimensions must be positive");

  // --- LHS check ---
  bool lhsIsMemref = (lhsVals.size() == 1) && mlir::isa<MemRefType>(lhsVals[0].getType());
  if (!lhsIsMemref) {
    if ((int64_t)lhsVals.size() != M * K)
      return emitOpError("lhs operand size does not match rowsA × colsA");
  } else {
    auto lhsType = mlir::cast<MemRefType>(lhsVals[0].getType());
    if (lhsType.hasStaticShape() && lhsType.getNumElements() < M * K)
      return emitOpError("lhs memref too small for rowsA × colsA");
  }

  // --- RHS check ---
  bool rhsIsMemref = (rhsVals.size() == 1) && mlir::isa<MemRefType>(rhsVals[0].getType());
  if (!rhsIsMemref) {
    if ((int64_t)rhsVals.size() != K * N)
      return emitOpError("rhs operand size does not match colsA × colsB");
  } else {
    auto rhsType = mlir::cast<MemRefType>(rhsVals[0].getType());
    if (rhsType.hasStaticShape() && rhsType.getNumElements() < K * N)
      return emitOpError("rhs memref too small for colsA × colsB");
  }

  // --- Output memref check ---
  if (outputType.hasStaticShape()) {
    int64_t outSize = outputType.getNumElements();
    if (outSize < M * N)
      return emitOpError("output memref too small for result matrix");
  }

  if (!outputType.getElementType().isInteger(8))
    return emitOpError("output must be memref of i8");

  // --- Value range check only for constant scalars ---
  if (!lhsIsMemref) {
    for (auto val : lhsVals)
      if (auto cst = val.getDefiningOp<arith::ConstantOp>())
        if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue()))
          if (intVal.getInt() < 0 || intVal.getInt() > 255)
            return emitOpError("lhs values must be in [0, 255]");
  }

  if (!rhsIsMemref) {
    for (auto val : rhsVals)
      if (auto cst = val.getDefiningOp<arith::ConstantOp>())
        if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue()))
          if (intVal.getInt() < 0 || intVal.getInt() > 255)
            return emitOpError("rhs values must be in [0, 255]");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SBoxOp
//===----------------------------------------------------------------------===//

LogicalResult SBoxOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue)
            return emitOpError("expects a constant integer input");
        if (intValue.getInt() < 0 || intValue.getInt() > 255)
            return emitOpError("input value must be in range [0, 255]");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// MixColumnsOp
//===----------------------------------------------------------------------===//
LogicalResult MixColumnsOp::verify() {
  // Get operands
  auto colMemRef = getCol();
  auto outMemRef = getOut();

  // Both must be memref<4xi8>
  auto colType = mlir::dyn_cast<MemRefType>(colMemRef.getType());
  auto outType = mlir::dyn_cast<MemRefType>(outMemRef.getType());

  if (!colType || !colType.getElementType().isInteger(8) || colType.getNumElements() != 4)
    return emitOpError("col operand must be memref<4xi8>");

  if (!outType || !outType.getElementType().isInteger(8) || outType.getNumElements() != 4)
    return emitOpError("out operand must be memref<4xi8>");

  for (auto val : getOperands()) {
    if (auto cst = val.getDefiningOp<arith::ConstantOp>())
      if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue()))
        if (intVal.getInt() < 0 || intVal.getInt() > 255)
          return emitOpError("all constant input values must be in [0, 255]");
  }

  return success();
}

