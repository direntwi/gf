# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own gf dialect.
add_mlir_aggregate(GFCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  GFCAPI
)

add_llvm_executable(gf-capi-test
  gf-capi-test.c
)
llvm_update_compile_flags(gf-capi-test)
target_link_libraries(gf-capi-test
  PRIVATE GFCAPITestLib)
