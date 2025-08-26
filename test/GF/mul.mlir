// module {
//   // An example function using gf.mul with non-zero arguments.
//   func.func @main() -> i8 {
//     %a = arith.constant 5 : i8
//     %b = arith.constant 7 : i8
//     %c = gf.mul %a, %b : i8
//     func.return %c : i8
//   }
// }

//For benchmarking
module {
  func.func @mul_stream_kernel(%x: memref<256xi8>, %y: memref<256xi8>, %z: memref<256xi8>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %iN = arith.constant 256 : index
    scf.for %i = %i0 to %iN step %i1 {
      %a = memref.load %x[%i] : memref<256xi8>
      %b = memref.load %y[%i] : memref<256xi8>
      %p = gf.mul %a, %b : i8
      memref.store %p, %z[%i] : memref<256xi8>
    }
    return
  }

  func.func @bench_mul_stream(%iters: index,
                            %x: memref<256xi8>, %y: memref<256xi8>, %z: memref<256xi8>) -> i8 {
  %c0    = arith.constant 0   : index
  %c1    = arith.constant 1   : index
  %i0    = arith.constant 0   : index
  %i1    = arith.constant 1   : index
  %i256  = arith.constant 256 : index
  %acc0  = arith.constant 0   : i8

  // iter_args carry: (acc, current X, current Z)
  %acc, %x_final, %z_final =
  scf.for %t = %c0 to %iters step %c1
      iter_args(%a = %acc0, %x_cur = %x, %z_cur = %z)
      -> (i8, memref<256xi8>, memref<256xi8>) {
    // 1) compute z_cur = x_cur (*) y
    func.call @mul_stream_kernel(%x_cur, %y, %z_cur)
      : (memref<256xi8>, memref<256xi8>, memref<256xi8>) -> ()

    // 2) per-iter consume: XOR-reduce z_cur into running accumulator
    %a_next = scf.for %j = %i0 to %i256 step %i1 iter_args(%cur = %a) -> i8 {
      %v  = memref.load %z_cur[%j] : memref<256xi8>
      %nx = arith.xori %cur, %v : i8
      scf.yield %nx : i8
    }

    // 3) ping-pong: next iter uses previous output as X; no memcpy
    scf.yield %a_next, %z_cur, %x_cur : i8, memref<256xi8>, memref<256xi8>
  }
  return %acc : i8
}




  func.func @main() -> i8 {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %iters = arith.constant 2000 : index

  %state1 = memref.alloca() : memref<256xi8>
  %state2 = memref.alloca() : memref<256xi8>
  %state3 = memref.alloca() : memref<256xi8>

  // init state1[i] = i, state2[i] = 3*i + 1  (mod 256)
  %i8_1 = arith.constant 1 : i8
  %i8_3 = arith.constant 3 : i8
  scf.for %i = %c0 to %c256 step %c1 {
    %i8  = arith.index_cast %i : index to i8
    memref.store %i8, %state1[%i] : memref<256xi8>
    %t0  = arith.muli %i8, %i8_3 : i8
    %t1  = arith.addi %t0, %i8_1 : i8
    memref.store %t1, %state2[%i] : memref<256xi8>
  }

  // run benchmark
  %acc = func.call @bench_mul_stream(%iters, %state1, %state2, %state3)
         : (index, memref<256xi8>, memref<256xi8>, memref<256xi8>) -> i8
return %acc : i8

}

}
