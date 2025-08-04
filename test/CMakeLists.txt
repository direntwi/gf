llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(GF_TEST_DEPENDS
        FileCheck count not
        gf-capi-test
        gf-opt
        gf-translate
        GFPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND GF_TEST_DEPENDS GFPythonModules)
endif()

add_lit_testsuite(check-gf "Running the gf regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${GF_TEST_DEPENDS}
        )
set_target_properties(check-gf PROPERTIES FOLDER "Tests")

add_lit_testsuites(GF ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${GF_TEST_DEPENDS})

add_subdirectory(CAPI)
