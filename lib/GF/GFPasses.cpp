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

    auto logSymAttr = SymbolRefAttr::get(rewriter.getContext(), "log_table");
    auto antiSymAttr = SymbolRefAttr::get(rewriter.getContext(), "antilog_table");

    // --- 2) Prepare constants
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);

    // --- 3) Fetch operands
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // --- 4) Early checks: zero or one
    Value lhsIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs, zero);
    Value rhsIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rhs, zero);
    Value eitherZero = rewriter.create<arith::OrIOp>(loc, lhsIsZero, rhsIsZero);

    Value lhsIsOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs, one);
    Value rhsIsOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rhs, one);

    // --- 5) Precompute result for early cases
    // If lhs == 1 -> rhs
    // Else if rhs == 1 -> lhs
    // Else 0
    Value partialResult = rewriter.create<arith::SelectOp>(
        loc, lhsIsOne, rhs,
        rewriter.create<arith::SelectOp>(
            loc, rhsIsOne, lhs, zero));

    // --- 6) Did we hit any early exit? (eitherZero OR lhsIsOne OR rhsIsOne)
    Value anyEarly1 = rewriter.create<arith::OrIOp>(loc, eitherZero, lhsIsOne);
    Value anyEarly = rewriter.create<arith::OrIOp>(loc, anyEarly1, rhsIsOne);

    // --- 7) Prepare index adjustment for log lookup (subtract 1)
    // (only used if anyEarly == false)
    Value oneI8 = one;
    Value lhsAdj = rewriter.create<arith::SubIOp>(loc, lhs, oneI8);
    Value rhsAdj = rewriter.create<arith::SubIOp>(loc, rhs, oneI8);

    Value lhsIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), lhsAdj);
    Value rhsIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), rhsAdj);

    // --- 8) Load tables
    auto i8Ty = rewriter.getIntegerType(8);
    auto logMemrefTy = MemRefType::get({255}, i8Ty);
    auto antiMemrefTy = MemRefType::get({255}, i8Ty);

    Value logTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, logMemrefTy, logSymAttr);
    Value antilogTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, antiMemrefTy, antiSymAttr);

    // --- 9) Conditionally load log values (safe dummy 0 if early)
    Value dummyLog = zero;
    Value logValLhs = rewriter.create<arith::SelectOp>(
        loc, anyEarly, dummyLog,
        rewriter.create<memref::LoadOp>(loc, logTablePtr, lhsIdx));
    Value logValRhs = rewriter.create<arith::SelectOp>(
        loc, anyEarly, dummyLog,
        rewriter.create<memref::LoadOp>(loc, logTablePtr, rhsIdx));

    // --- 10) Sum logs and mod 255
    Value logSum = rewriter.create<arith::AddIOp>(loc, logValLhs, logValRhs);
    Value modConst = rewriter.create<arith::ConstantIntOp>(loc, 255, 8);
    Value modSum = rewriter.create<arith::RemUIOp>(loc, logSum, modConst);

    // --- 11) Index cast for antilog lookup
    Value modIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), modSum);

    // --- 12) Load antilog value (safe dummy 0 if early)
    Value prodVal = rewriter.create<arith::SelectOp>(
        loc, anyEarly, zero,
        rewriter.create<memref::LoadOp>(loc, antilogTablePtr, modIdx));

    // --- 13) If early, return partialResult; else return prodVal
    Value finalResult = rewriter.create<arith::SelectOp>(
        loc, anyEarly, partialResult, prodVal);

    // --- 14) Replace op
    rewriter.replaceOp(op, finalResult);
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

    auto logSymAttr = SymbolRefAttr::get(rewriter.getContext(), "log_table");
    auto antiSymAttr = SymbolRefAttr::get(rewriter.getContext(), "antilog_table");

    // --- 2) Constants
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
    Value c255 = rewriter.create<arith::ConstantIntOp>(loc, 255, 8);

    // --- 3) Zero check
    Value in = op.getOperand();
    Value isZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, in, zero);

    // --- 4) Adjust index for log lookup (subtract 1)
    Value inAdj = rewriter.create<arith::SubIOp>(loc, in, one);
    Value inIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), inAdj);

    // --- 5) Load from log_table
    auto i8Ty = rewriter.getIntegerType(8);
    auto logMemrefTy = MemRefType::get({255}, i8Ty);
    Value logTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, logMemrefTy, logSymAttr);
    Value logVal = rewriter.create<memref::LoadOp>(loc, logTablePtr, inIdx);

    // --- 6) Compute (255 - logVal) mod 255
    Value diff = rewriter.create<arith::SubIOp>(loc, c255, logVal);
    Value invIdxI8 = rewriter.create<arith::RemUIOp>(loc, diff, c255);
    Value invIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), invIdxI8);

    // --- 7) Load from antilog_table
    auto antiMemrefTy = MemRefType::get({255}, i8Ty);
    Value antiTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, antiMemrefTy, antiSymAttr);
    Value invVal = rewriter.create<memref::LoadOp>(loc, antiTablePtr, invIdx);

    // --- 8) Select: if zero, return zero, else return invVal
    Value result = rewriter.create<arith::SelectOp>(
        loc, isZero, zero, invVal);

    // --- 9) Replace
    rewriter.replaceOp(op, result);
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
        /*output=*/outMemRef,
        rewriter.getI8IntegerAttr(4),
        rewriter.getI8IntegerAttr(4),
        rewriter.getI8IntegerAttr(1));

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
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<
    GFAddOpLowering, 
    GFMulOpLowering,
    GFInvOpLowering,
    GFMatMulOpLowering,
    GFSBoxOpLowering,
    GFMixColumnsOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGFToArithPass() { 
  return std::make_unique<ConvertGFToArithPass>();
}

} // namespace mlir::gf
