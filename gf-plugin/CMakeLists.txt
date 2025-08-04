add_llvm_library(GFPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        gf-plugin.cpp

        DEPENDS
        MLIRGF
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRGF
        )
target_include_directories(
        GFPlugin
        PRIVATE
        "${GF_BINARY_DIR}/include"
)
