// RUN: gf-opt %s | gf-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = gf.foo %{{.*}} : i32
        %res = gf.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @gf_types(%arg0: !gf.custom<"10">)
    func.func @gf_types(%arg0: !gf.custom<"10">) {
        return
    }
}
