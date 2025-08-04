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
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "GF/GFPasses.h"

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


struct ConvertGFToArithPass 
    : public PassWrapper<ConvertGFToArithPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "convert-gf-to-arith"; }
  StringRef getDescription() const final { 
    return "Convert GF ops to Arith dialect operations"; 
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GFAddOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGFToArithPass() { 
  return std::make_unique<ConvertGFToArithPass>();
}

} // namespace mlir::gf
