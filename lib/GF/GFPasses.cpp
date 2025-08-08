//===- GFPasses.cpp - GF passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "GF/GFPasses.h"
#include "GF/GFUtils.h"

namespace mlir::gf {
#define GEN_PASS_DEF_GFSWITCHBARFOO
#include "GF/GFPasses.h.inc"

namespace {
class GFSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class GFSwitchBarFoo
    : public impl::GFSwitchBarFooBase<GFSwitchBarFoo> {
public:
  using impl::GFSwitchBarFooBase<
      GFSwitchBarFoo>::GFSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<GFSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

struct GFAddOpLowering : public OpRewritePattern<gf::AddOp> {
  using OpRewritePattern<gf::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::AddOp op, 
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Location loc = op.getLoc();

    // XOR the two operands
    Value result = rewriter.create<arith::XOrIOp>(loc, lhs, rhs);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

struct GFMulOpLowering : public OpRewritePattern<gf::MulOp> {
    using OpRewritePattern<gf::MulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gf::MulOp op,
                                  PatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      auto module = op->getParentOfType<ModuleOp>();
      if (!module)
        return rewriter.notifyMatchFailure(op, "not in a module");

      // --- 1) Ensure the log/antilog globals have been injected into the module.
      using GlobalOp = memref::GlobalOp;

      // Only inject if the module doesn’t already have a GlobalOp named “log_table”
      if (!module.lookupSymbol<GlobalOp>("log_table")) {
        auto lookupM = parseSourceString<ModuleOp>(kLogAntilogTables, rewriter.getContext());
        if (!lookupM) return failure();
        SymbolTable symTable(module);
        // 1) Clone all memref.global @log_table / @antilog_table
        for (auto glob : lookupM->getOps<GlobalOp>()) {
          if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPointToEnd(module.getBody());
            rewriter.clone(*glob.getOperation());
          }
        }
      }

     // --- Fetch and widen operands to i32
    Value lhs8 = op.getLhs();
    Value rhs8 = op.getRhs();
    Value lhs32 = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), lhs8);
    Value rhs32 = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), rhs8);

    // --- Constants in i32
    Value zero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value one32  = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

    // --- Early checks in i32
    Value lhsIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs32, zero32);
    Value rhsIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rhs32, zero32);
    Value eitherZero = rewriter.create<arith::OrIOp>(loc, lhsIsZero, rhsIsZero);
    Value lhsIsOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs32, one32);
    Value rhsIsOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rhs32, one32);

    // --- Partial result for early cases (i32)
    Value partialResult32 = rewriter.create<arith::SelectOp>(
        loc, lhsIsOne, rhs32,
        rewriter.create<arith::SelectOp>(loc, rhsIsOne, lhs32, zero32));

    Value anyEarly1 = rewriter.create<arith::OrIOp>(loc, eitherZero, lhsIsOne);
    Value anyEarly  = rewriter.create<arith::OrIOp>(loc, anyEarly1, rhsIsOne);

    // --- Adjust indices for table lookup (i32)
    Value lhsAdj = rewriter.create<arith::SubIOp>(loc, lhs32, one32);
    Value rhsAdj = rewriter.create<arith::SubIOp>(loc, rhs32, one32);
    Value lhsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), lhsAdj);
    Value rhsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), rhsAdj);

    // --- Load tables (i8) and widen to i32
    auto i8Ty = rewriter.getIntegerType(8);
    Value logTable = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({255}, i8Ty),
        SymbolRefAttr::get(rewriter.getContext(), "log_table"));
    Value antiTable = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({255}, i8Ty),
        SymbolRefAttr::get(rewriter.getContext(), "antilog_table"));

    Value dummyLog8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value logVal8L = rewriter.create<arith::SelectOp>(
        loc, anyEarly,
        dummyLog8,
        rewriter.create<memref::LoadOp>(loc, logTable, lhsIdx));
    Value logVal8R = rewriter.create<arith::SelectOp>(
        loc, anyEarly,
        dummyLog8,
        rewriter.create<memref::LoadOp>(loc, logTable, rhsIdx));
    Value logValL = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), logVal8L);
  Value logValR = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), logVal8R);

    // --- Sum logs and mod 255 (i32)
    Value logSum = rewriter.create<arith::AddIOp>(loc, logValL, logValR);
    Value modConst32 = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);
    Value modSum = rewriter.create<arith::RemUIOp>(loc, logSum, modConst32);

    // --- Antilog lookup
    Value modIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), modSum);
    Value prodVal8 = rewriter.create<arith::SelectOp>(
        loc, anyEarly,
        dummyLog8,
        rewriter.create<memref::LoadOp>(loc, antiTable, modIdx));
    Value prodVal32 = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), prodVal8);

    // --- Final select and truncate to i8
    Value final32 = rewriter.create<arith::SelectOp>(loc, anyEarly,
                                                   partialResult32, prodVal32);
    Value final8 = rewriter.create<arith::TruncIOp>(loc, rewriter.getI8Type(), final32);

    rewriter.replaceOp(op, final8);
    return success();
    }
  };

//===----------------------------------------------------------------------===//
// InvOp
//===----------------------------------------------------------------------===//

struct GFInvOpLowering : public OpRewritePattern<gf::InvOp> {
  using OpRewritePattern<gf::InvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::InvOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    using GlobalOp = memref::GlobalOp;

    // 1) Inject lookup‑table funcs if missing
    if (!module.lookupSymbol<GlobalOp>("log_table")) {
      auto lookupM = parseSourceString<ModuleOp>(
          kLogAntilogTables, rewriter.getContext());
      if (!lookupM) return failure();
      SymbolTable symtab(module);
      for (auto glob : lookupM->getOps<GlobalOp>()) {
        if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*glob.getOperation());
        }
      }
    }

    // --- 2) Load input and widen to i32
    Value in8   = op.getOperand();
    Value in32  = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), in8);

    // --- 3) Constants in i32
    Value zero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value one32  = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value c255_32 = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);

    // --- 4) Zero check
    Value isZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, in32, zero32);

    // --- 5) Compute index = (255 - logVal) mod 255
    // Subtract 1 for table index: idx = in32 - 1
    Value inAdj32 = rewriter.create<arith::SubIOp>(loc, in32, one32);
    // Cast to Index for table load
    Value inIdx = rewriter.create<arith::IndexCastOp>(loc,
        rewriter.getIndexType(), inAdj32);

    // Load log table entry (i8) and widen
    auto i8Ty = rewriter.getIntegerType(8);
    Value logTable = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({255}, i8Ty),
        SymbolRefAttr::get(rewriter.getContext(), "log_table"));
    Value log8 = rewriter.create<memref::LoadOp>(loc, logTable, inIdx);
    Value log32 = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), log8);

    // Compute diff = (255 - log32)
    Value diff32 = rewriter.create<arith::SubIOp>(loc, c255_32, log32);
    // Mod 255: invIdx32 = diff32 % 255
    Value invIdx32 = rewriter.create<arith::RemUIOp>(loc, diff32, c255_32);
    // Cast to Index for antilog lookup
    Value invIdx = rewriter.create<arith::IndexCastOp>(loc,
        rewriter.getIndexType(), invIdx32);

    // Load anti-log entry (i8) and widen to i32
    Value antiTable = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({255}, i8Ty),
        SymbolRefAttr::get(rewriter.getContext(), "antilog_table"));
    Value inv8 = rewriter.create<memref::LoadOp>(loc, antiTable, invIdx);
    Value inv32 = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), inv8);

    // --- 6) Select: if input was zero, result = zero; else result = inv32
    Value result32 = rewriter.create<arith::SelectOp>(
        loc, isZero, zero32, inv32);

    // --- 7) Truncate back to i8 and replace
    Value result8 = rewriter.create<arith::TruncIOp>(loc,
        rewriter.getI8Type(), result32);
    rewriter.replaceOp(op, result8);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

struct GFMatMulOpLowering : OpRewritePattern<gf::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::MatMulOp op,
                              PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    int64_t M = op.getRowsA();
    int64_t K = op.getColsA();
    int64_t N = op.getColsB();
    auto operands = op.getOperands();

     // --- LHS ---
    bool lhsIsMemref = mlir::isa<MemRefType>(operands[0].getType());
    int64_t numLhs = lhsIsMemref ? 1 : (M * K);
    Value memrefA = lhsIsMemref
        ? operands[0]
        : gf::materializeMemref(loc, rewriter,
              SmallVector<Value>(operands.begin(), operands.begin() + numLhs));

    // --- RHS ---
    bool rhsIsMemref = mlir::isa<MemRefType>(operands[numLhs].getType());
    int64_t numRhs = rhsIsMemref ? 1 : (K * N);
    Value memrefB = rhsIsMemref
        ? operands[numLhs]
        : gf::materializeMemref(loc, rewriter,
              SmallVector<Value>(operands.begin() + numLhs, operands.begin() + numLhs + numRhs));

    // --- Output ---
    Value outputMemRef = operands[numLhs + numRhs];

    // Loop constants
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cK = rewriter.create<arith::ConstantIndexOp>(loc, K);
    auto cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);

    // Loop nest: i and j
    for (int64_t i = 0; i < M; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);

      for (int64_t j = 0; j < N; ++j) {
        Value jVal = rewriter.create<arith::ConstantIndexOp>(loc, j);

        // k-loop
        auto loop = rewriter.create<scf::ForOp>(loc, c0, cK, c1, ValueRange{zero});
        rewriter.setInsertionPointToStart(loop.getBody());

        Value k = loop.getInductionVar();
        Value acc = loop.getRegionIterArgs()[0];

        // A[i*K + k]
        Value aIdx = rewriter.create<arith::AddIOp>(
            loc, rewriter.create<arith::MulIOp>(loc, iVal, cK), k);
        Value lhs = rewriter.create<memref::LoadOp>(loc, memrefA, aIdx);

        // B[k*N + j]
        Value bIdx = rewriter.create<arith::AddIOp>(
            loc, rewriter.create<arith::MulIOp>(loc, k, cN), jVal);
        Value rhs = rewriter.create<memref::LoadOp>(loc, memrefB, bIdx);

        // Multiply in GF(2^8) and XOR accumulate
        Value prod = rewriter.create<gf::MulOp>(loc, lhs, rhs);
        Value newAcc = rewriter.create<arith::XOrIOp>(loc, acc, prod);
        rewriter.create<scf::YieldOp>(loc, newAcc);

        // After loop
        rewriter.setInsertionPointAfter(loop);
        Value result = loop.getResult(0);
        Value outIdx = rewriter.create<arith::AddIOp>(
            loc, rewriter.create<arith::MulIOp>(loc, iVal, cN), jVal);
        rewriter.create<memref::StoreOp>(loc, result, outputMemRef, outIdx);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SBoxOp
//===----------------------------------------------------------------------===//

struct GFSBoxOpLowering : public OpRewritePattern<gf::SBoxOp> {
  using OpRewritePattern<gf::SBoxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::SBoxOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // 1) Ensure sbox_table is in the module:
    using GlobalOp = memref::GlobalOp;

    // 1) Inject lookup‑table if missing
    if (!module.lookupSymbol<GlobalOp>("sbox_table")) {
      auto lookupM = parseSourceString<ModuleOp>(
          kSBoxLookupTable, rewriter.getContext());
      if (!lookupM) return failure();
      SymbolTable symtab(module);
      for (auto glob : lookupM->getOps<GlobalOp>()) {
        if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*glob.getOperation());
        }
      }
    }

    // 2) Get the memref:
    Value tableMemref = rewriter.create<memref::GetGlobalOp>(
      loc,
      MemRefType::get({256}, rewriter.getI8Type()),
      "sbox_table"
    );

    // 3) Cast the input byte to index:
    Value idx = rewriter.create<arith::IndexCastOp>(
      loc,
      rewriter.getIndexType(),
      op.getInput()
    );

    // 4) Load the substituted value:
    Value out = rewriter.create<memref::LoadOp>(
      loc,
      tableMemref,
      idx
    );

    rewriter.replaceOp(op, out);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MixColumnsOp
//===----------------------------------------------------------------------===//

struct GFMixColumnsOpLowering : public OpRewritePattern<gf::MixColumnsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::MixColumnsOp op,
                              PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value colMemRef = op.getCol();
    Value outMemRef = op.getOut();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    using GlobalOp = memref::GlobalOp;

    if (!module.lookupSymbol<GlobalOp>("aes_mix_columns_matrix")) {
      auto matrixModule = parseSourceString<ModuleOp>(
          kAESMixColumnsMatrix, rewriter.getContext());
      if (!matrixModule) return failure();
      SymbolTable symtab(module);
      for (auto glob : matrixModule->getOps<GlobalOp>()) {
        if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*glob.getOperation());
        }
      }
    }

    // Load matrix reference
    Value mat = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({16}, rewriter.getI8Type()), "aes_mix_columns_matrix");
    
    // Allocate temporary buffer for MatMul result (4 elements)
    auto tmpType = MemRefType::get({4}, rewriter.getI8Type());
    Value tmpResult = rewriter.create<memref::AllocaOp>(loc, tmpType);

    // Load column values
    SmallVector<Value> colValues;
    for (int i = 0; i < 4; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      colValues.push_back(rewriter.create<memref::LoadOp>(loc, colMemRef, idx));
    }

    // Call MatMul
    rewriter.create<gf::MatMulOp>(
        loc,
        /*lhs=*/ValueRange{mat},
        /*rhs=*/colValues,
        /*output=*/tmpResult,
        rewriter.getI8IntegerAttr(4),
        rewriter.getI8IntegerAttr(4),
        rewriter.getI8IntegerAttr(1));

    // Mask and copy results into output memref
    Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xFF, 8);
    for (int i = 0; i < 4; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value val = rewriter.create<memref::LoadOp>(loc, tmpResult, idx);
      Value masked = rewriter.create<arith::AndIOp>(loc, val, mask);
      rewriter.create<memref::StoreOp>(loc, masked, outMemRef, idx);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// KeyScheduleOp
//===----------------------------------------------------------------------===//

struct GFKeyScheduleOpLowering : public OpRewritePattern<gf::KeyScheduleOp> {
  using OpRewritePattern<gf::KeyScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::KeyScheduleOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value key = op.getKey();
    Value schedule = op.getSchedule();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    // Step 1: Copy initial key bytes (i8) directly
    for (int i = 0; i < 16; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value byte = rewriter.create<memref::LoadOp>(loc, key, idx);
      rewriter.create<memref::StoreOp>(loc, byte, schedule, idx);
    }

    // Step 2: Ensure RCON global exists
    using GlobalOp = memref::GlobalOp;
    if (!module.lookupSymbol<GlobalOp>("aes_rcon")) {
      auto rconMod = parseSourceString<ModuleOp>(kAESRcon, rewriter.getContext());
      if (!rconMod) return failure();
      SymbolTable symtab(module);
      for (auto glob : rconMod->getOps<GlobalOp>()) {
        if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*glob.getOperation());
        }
      }
    }
    Value rconMem = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({11}, rewriter.getI8Type()), "aes_rcon");

    Value c4  = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value c44 = rewriter.create<arith::ConstantIndexOp>(loc, 44);
    Value c1  = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Step 3: Loop for words 4..43
    rewriter.create<scf::ForOp>(
        loc, c4, c44, c1, ValueRange{},
        [&](OpBuilder &b, Location l, Value iv, ValueRange) {
          // Compute base index = iv * 4
          Value base = b.create<arith::MulIOp>(l, iv,
                           b.create<arith::ConstantIndexOp>(l, 4));

          // Load previous words and widen to i32
          SmallVector<Value> prev(4), prev4(4);
          for (int j = 0; j < 4; ++j) {
            Value idxPrev = b.create<arith::AddIOp>(l, base,
                               b.create<arith::ConstantIndexOp>(l, j - 4));
            Value v8Prev  = b.create<memref::LoadOp>(l, schedule, idxPrev);
            prev[j]       = b.create<arith::ExtUIOp>(l, b.getI32Type(), v8Prev);

            Value idxPrev4 = b.create<arith::AddIOp>(l, base,
                                b.create<arith::ConstantIndexOp>(l, j - 16));
            Value v8Prev4  = b.create<memref::LoadOp>(l, schedule, idxPrev4);
            prev4[j]      = b.create<arith::ExtUIOp>(l, b.getI32Type(), v8Prev4);
          }
          SmallVector<Value> temp = prev;

          // Condition: iv % 4 == 0
          Value rem = b.create<arith::RemUIOp>(l, iv,
                         b.create<arith::ConstantIndexOp>(l, 4));
          Value doRcon = b.create<arith::CmpIOp>(
                             l, arith::CmpIPredicate::eq, rem,
                             b.create<arith::ConstantIndexOp>(l, 0));

          // IF for RotWord+SubWord+RCON
          auto ifOp = b.create<scf::IfOp>(
              l, SmallVector<Type>(4, b.getI32Type()), doRcon, true);

          // THEN branch
          {
            OpBuilder thenB = ifOp.getThenBodyBuilder();
            // Rotate
            SmallVector<Value> rot = {temp[1], temp[2], temp[3], temp[0]};
            for (int k = 0; k < 4; ++k) {
              // SBox returns i8, extend to i32
              Value s8 = thenB.create<gf::SBoxOp>(l, rot[k]);
              rot[k] = thenB.create<arith::ExtUIOp>(l, thenB.getI32Type(), s8);
            }
            // RCON
            Value div4 = thenB.create<arith::DivUIOp>(l, iv,
                             thenB.create<arith::ConstantIndexOp>(l, 4));
            Value rc8 = thenB.create<memref::LoadOp>(l, rconMem, div4);
            Value rc32 = thenB.create<arith::ExtUIOp>(l, thenB.getI32Type(), rc8);
            // XOR
            rot[0] = thenB.create<arith::XOrIOp>(l, rot[0], rc32);
            thenB.create<scf::YieldOp>(l, rot);
          }
          // ELSE branch
          {
            OpBuilder elseB = ifOp.getElseBodyBuilder();
            elseB.create<scf::YieldOp>(l, temp);
          }
          // Update temp
          temp.assign(ifOp.getResults().begin(), ifOp.getResults().end());

          // Compute newWord = temp ^ prev4
          SmallVector<Value> newWord(4);
          for (int k = 0; k < 4; ++k) {
            newWord[k] = b.create<arith::XOrIOp>(l, temp[k], prev4[k]);
          }

          // Store newWord truncated to i8
          for (int k = 0; k < 4; ++k) {
            Value idx = b.create<arith::AddIOp>(l, base,
                b.create<arith::ConstantIndexOp>(l, k));
            Value t8  = b.create<arith::TruncIOp>(l, b.getI8Type(), newWord[k]);
            b.create<memref::StoreOp>(l, t8, schedule, idx);
          }
          b.create<scf::YieldOp>(l);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddRoundKeyOp
//===----------------------------------------------------------------------===//

struct GFAddRoundKeyOpLowering : public OpRewritePattern<gf::AddRoundKeyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::AddRoundKeyOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value state = op.getState();
    Value roundKey = op.getRoundKey();

    // 1. Define loop bounds
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value sixteen = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // 2. Generate a loop over [0, 16)
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, zero, sixteen, one, std::nullopt,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange /*args*/) {
          // state[i]
          Value sByte = b.create<memref::LoadOp>(loc, state, iv);
          // roundKey[i]
          Value kByte = b.create<memref::LoadOp>(loc, roundKey, iv);

          // XOR using gf::AddOp
          Value xored = b.create<gf::AddOp>(loc, sByte, kByte);

          // store back into state[i]
          b.create<memref::StoreOp>(loc, xored, state, iv);

          b.create<scf::YieldOp>(loc);
        });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ShiftRowsOp
//===----------------------------------------------------------------------===//

struct GFShiftRowsOpLowering : OpRewritePattern<gf::ShiftRowsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gf::ShiftRowsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value state = op.getState();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    using GlobalOp = memref::GlobalOp;
      if (!module.lookupSymbol<GlobalOp>("aes_shift_rows_matrix")) {
        auto shiftmap = parseSourceString<ModuleOp>(kAESShiftRowsMatrix, rewriter.getContext());
        if (!shiftmap) return failure();
        
        SymbolTable symtab(module);
        for (auto glob : shiftmap->getOps<GlobalOp>()) {
          if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(module.getBody());
            rewriter.clone(*glob.getOperation());
          }
        }
      }


    // Allocate temporary buffer: memref<16xi8>
    auto memrefType = mlir::dyn_cast<MemRefType>(state.getType());
    Value temp = rewriter.create<memref::AllocOp>(loc, memrefType);

    // Load the global shift map
    Value lut = rewriter.create<memref::GetGlobalOp>(
        loc,
        MemRefType::get({16}, rewriter.getI8Type()),
        StringAttr::get(rewriter.getContext(), "aes_shift_rows_matrix"));

    // Perform state[i] = state[shiftmap[i]]
    for (int i = 0; i < 16; ++i) {
      Value iC = rewriter.create<arith::ConstantIndexOp>(loc, i);

      // mappedIndex = shiftmap[i]
      Value mappedIndex = rewriter.create<memref::LoadOp>(loc, lut, iC);

      // mappedIndex : i8 → index
      Value mappedIndexIdx = rewriter.create<arith::IndexCastUIOp>(
          loc, rewriter.getIndexType(), mappedIndex);

      // val = state[shiftmap[i]]
      Value val = rewriter.create<memref::LoadOp>(loc, state, mappedIndexIdx);

      // temp[i] = val
      rewriter.create<memref::StoreOp>(loc, val, temp, iC);
    }

    // Copy temp back to state
    for (int i = 0; i < 16; ++i) {
      Value iC = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value val = rewriter.create<memref::LoadOp>(loc, temp, iC);
      rewriter.create<memref::StoreOp>(loc, val, state, iC);
    }

    // Dealloc temp
    rewriter.create<memref::DeallocOp>(loc, temp);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// ConvertGFToArithPass
//===----------------------------------------------------------------------===//


struct ConvertGFToArithPass 
    : public PassWrapper<ConvertGFToArithPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "convert-gf-to-arith"; }
  StringRef getDescription() const final { 
    return "Convert GF ops to Arith dialect operations"; 
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<gf::AddOp>();
    target.addIllegalOp<gf::MulOp>();
    target.addIllegalOp<gf::InvOp>();
    target.addIllegalOp<gf::MatMulOp>();
    target.addIllegalOp<gf::SBoxOp>();
    target.addIllegalOp<gf::MixColumnsOp>();
    target.addIllegalOp<gf::KeyScheduleOp>();
    target.addIllegalOp<gf::AddRoundKeyOp>();
    target.addIllegalOp<gf::ShiftRowsOp>();
    target.addIllegalDialect<gf::GFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<
    GFAddOpLowering, 
    GFMulOpLowering,
    GFInvOpLowering,
    GFMatMulOpLowering,
    GFSBoxOpLowering,
    GFMixColumnsOpLowering,
    GFKeyScheduleOpLowering,
    GFAddRoundKeyOpLowering,
    GFShiftRowsOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet nestedPatterns(&getContext());
    nestedPatterns.add<GFSBoxOpLowering>(&getContext());
    ConversionTarget nestedTarget(getContext());
    nestedTarget.addIllegalOp<gf::SBoxOp>();
    nestedTarget.addLegalDialect<arith::ArithDialect>();
    nestedTarget.addLegalDialect<memref::MemRefDialect>();
    nestedTarget.addLegalDialect<scf::SCFDialect>();

    if (failed(applyPartialConversion(getOperation(), nestedTarget, std::move(nestedPatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGFToArithPass() { 
  return std::make_unique<ConvertGFToArithPass>();
}

} // namespace mlir::gf
