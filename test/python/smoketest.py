# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_gf.ir import *
from mlir_gf.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_gf.dialects import gf_pybind11 as gf_d
elif sys.argv[1] == "nanobind":
    from mlir_gf.dialects import gf_nanobind as gf_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    gf_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = gf.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: gf.foo %[[C]] : i32
    print(str(module))
