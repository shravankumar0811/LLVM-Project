// RUN: mlir-opt %s | FileCheck %s

func.func @main() {
    "hello.world"() : () -> ()
    return
}

// CHECK: hello.world