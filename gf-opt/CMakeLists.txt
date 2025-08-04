get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRGF
        MLIRConvertToLLVMPass
        )
add_llvm_executable(gf-opt gf-opt.cpp)

llvm_update_compile_flags(gf-opt)
target_link_libraries(gf-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(gf-opt)
