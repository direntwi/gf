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
        MLIRFuncAllExtensions
        MLIRTensorAllExtensions #only added because we registered all extensions in gf-opt.cpp
        )
