add_mlir_dialect_library(MLIRGF
        GFTypes.cpp
        GFDialect.cpp
        GFOps.cpp
        GFPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/GF

        DEPENDS
        MLIRGFOpsIncGen
        MLIRGFPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
