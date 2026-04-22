//===- GFOps.cpp - GF dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GF/GFOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"


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
        if (!intValue){
          return emitOpError("expects a constant integer input");
        }
            
        if (intValue.getInt() < 0 || intValue.getInt() > 255){
          return emitOpError("input value must be in range [0, 255]");
        }
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

  int64_t m = getRowsA();
  int64_t k = getColsA();
  int64_t n = getColsB();

  // Check dimensions are positive
  if (m <= 0 || k <= 0 || n <= 0) {
    return emitOpError("all matrix dimensions must be positive");
  }

  // --- LHS check ---
  bool lhsIsMemref = (lhsVals.size() == 1) && mlir::isa<MemRefType>(lhsVals[0].getType());
  if (!lhsIsMemref) {
    if ((int64_t)lhsVals.size() != m * k) {
      return emitOpError("lhs operand size does not match rowsA × colsA");
    }
  } 
  else {
    auto lhsType = mlir::cast<MemRefType>(lhsVals[0].getType());
    if (lhsType.hasStaticShape() && lhsType.getNumElements() < m * k){
      return emitOpError("lhs memref too small for rowsA × colsA");
    }
  }

  // --- RHS check ---
  bool rhsIsMemref = (rhsVals.size() == 1) && mlir::isa<MemRefType>(rhsVals[0].getType());
  if (!rhsIsMemref) {
    if ((int64_t)rhsVals.size() != k * n) {
      return emitOpError("rhs operand size does not match colsA × colsB");
}
  } else {
    auto rhsType = mlir::cast<MemRefType>(rhsVals[0].getType());
    if (rhsType.hasStaticShape() && rhsType.getNumElements() < k * n) {
      return emitOpError("rhs memref too small for colsA × colsB");
}
  }

  // --- Output memref check ---
  if (outputType.hasStaticShape()) {
    int64_t outSize = outputType.getNumElements();
    if (outSize < m * n) {
      return emitOpError("output memref too small for result matrix");
    } 
  }

  if (!outputType.getElementType().isInteger(8)) {
    return emitOpError("output must be memref of i8");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SBoxOp
//===----------------------------------------------------------------------===//

LogicalResult SBoxOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue) {
            return emitOpError("expects a constant integer input");
        } 
        if (intValue.getInt() < 0 || intValue.getInt() > 255) {
            return emitOpError("input value must be in range [0, 255]");
        }
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

  if (!colType || !colType.getElementType().isInteger(8) || colType.getNumElements() != 4) {
    return emitOpError("col operand must be memref<4xi8>");
  }

  if (!outType || !outType.getElementType().isInteger(8) || outType.getNumElements() != 4) {
    return emitOpError("out operand must be memref<4xi8>");
  }

  for (auto val : getOperands()) {
    if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
      if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue())) {
        if (intVal.getInt() < 0 || intVal.getInt() > 255) {
          return emitOpError("all constant input values must be in [0, 255]");
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
//KeyScheduleOp
//===----------------------------------------------------------------------===//

LogicalResult KeyScheduleOp::verify() {
  auto keyType = dyn_cast<MemRefType>(getKey().getType());
  auto schedType = dyn_cast<MemRefType>(getSchedule().getType());

  if (!keyType || !schedType) {
    return emitOpError("expected memref types for key and schedule");
  }

  if (keyType.getShape().size() != 1 || keyType.getShape()[0] != 16) {
    return emitOpError("key must be memref<16xi8>");
  }

  if (schedType.getShape().size() != 1 || schedType.getShape()[0] != 176) {
    return emitOpError("schedule must be memref<176xi8>");
  } 

  if (!keyType.getElementType().isInteger(8) || !schedType.getElementType().isInteger(8)) {
    return emitOpError("key and schedule must have element type i8");
  }

  if (getKey() == getSchedule()) {
    return emitOpError("key and schedule cannot be the same memref");
  }


  for (auto val : getOperands()) {
    if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
      if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue())) {
        if (intVal.getInt() < 0 || intVal.getInt() > 255) {
          return emitOpError("all constant input values must be in [0, 255]");
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AddRoundKeyOp
//===----------------------------------------------------------------------===//  

LogicalResult AddRoundKeyOp::verify() {
  // Get operands
  auto state = getState();
  auto roundKey = getRoundKey();

  // Both must be memref<16xi8>
  auto stateType = mlir::dyn_cast<MemRefType>(state.getType());
  auto roundKeyType = mlir::dyn_cast<MemRefType>(roundKey.getType());

  if (stateType.getShape().size() != 1 || stateType.getShape()[0] != 16) {
    return emitOpError("state must be memref<16xi8>");
  }

  if (roundKeyType.getShape().size() != 1 || roundKeyType.getShape()[0] != 16) {
    return emitOpError("round key must be memref<16xi8>");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShiftRowsOp
//===----------------------------------------------------------------------===//

LogicalResult ShiftRowsOp::verify() {
  auto state = getState();
  auto stateType = mlir::dyn_cast<MemRefType>(state.getType());
  
  if (stateType.getShape().size() != 1 || stateType.getShape()[0] != 16) {
    return emitOpError("state must be memref<16xi8>");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InvSBoxOp
//===----------------------------------------------------------------------===//

LogicalResult InvSBoxOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
      auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!intValue) {
          return emitOpError("expects a constant integer input");
      }
      if (intValue.getInt() < 0 || intValue.getInt() > 255) {
          return emitOpError("input value must be in range [0, 255]");
      }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// InvShiftRowsOp
//===----------------------------------------------------------------------===//

LogicalResult InvShiftRowsOp::verify() {
  auto state = getState();
  auto stateType = mlir::dyn_cast<MemRefType>(state.getType());
  
  if (stateType.getShape().size() != 1 || stateType.getShape()[0] != 16) {
    return emitOpError("state must be memref<16xi8>");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InvMixColumnsOp
//===----------------------------------------------------------------------===//
LogicalResult InvMixColumnsOp::verify() {

  auto colMemRef = getCol();
  auto outMemRef = getOut();

  auto colType = mlir::dyn_cast<MemRefType>(colMemRef.getType());
  auto outType = mlir::dyn_cast<MemRefType>(outMemRef.getType());

  if (!colType || !colType.getElementType().isInteger(8) || colType.getNumElements() != 4) {
    return emitOpError("col operand must be memref<4xi8>");
  }

  if (!outType || !outType.getElementType().isInteger(8) || outType.getNumElements() != 4) {
    return emitOpError("out operand must be memref<4xi8>");
  }

  return success();
}