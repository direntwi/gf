include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_gf`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_gf.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(GFPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT GFPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_gf"
  TD_FILE dialects/GFOps.td
  SOURCES
    dialects/gf_pybind11.py
    dialects/gf_nanobind.py
  DIALECT_NAME gf)


declare_mlir_python_extension(GFPythonSources.Pybind11Extension
  MODULE_NAME _gfDialectsPybind11
  ADD_TO_PARENT GFPythonSources
  SOURCES
    GFExtensionPybind11.cpp
  EMBED_CAPI_LINK_LIBS
    GFCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(GFPythonSources.NanobindExtension
  MODULE_NAME _gfDialectsNanobind
  ADD_TO_PARENT GFPythonSources
  SOURCES
    GFExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    GFCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(GFPythonCAPI
  INSTALL_COMPONENT GFPythonModules
  INSTALL_DESTINATION python_packages/gf/mlir_gf/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/gf/mlir_gf/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    GFPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(GFPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/gf/mlir_gf"
  INSTALL_PREFIX "python_packages/gf/mlir_gf"
  DECLARED_SOURCES
    GFPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    GFPythonCAPI
  )
