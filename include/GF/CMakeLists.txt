add_mlir_dialect(GFOps gf)
add_mlir_doc(GFDialect GFDialect GF/ -gen-dialect-doc)
add_mlir_doc(GFOps GFOps GF/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GFPasses.td)
mlir_tablegen(GFPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRGFPassesIncGen)
