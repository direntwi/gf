// RUN: mlir-opt %s --load-dialect-plugin=%gf_libs/GFPlugin%shlibext --pass-pipeline="builtin.module(gf-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @gf_types(%arg0: !gf.custom<"10">)
  func.func @gf_types(%arg0: !gf.custom<"10">) {
    return
  }
}
