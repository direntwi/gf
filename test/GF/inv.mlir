// module {
//   // An example function using gf.inv with a non-zero argument.
//   func.func @main() -> i8 {
//     %a = arith.constant 5 : i8
//     %c = gf.inv %a : i8
//     func.return %c : i8
//   }
// }


//Benchmarking:
module {
  // Kernel: out[i] = inv(in[i]) for 256 bytes
  func.func @inv_domain_kernel(%in: memref<256xi8>, %out: memref<256xi8>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %iN = arith.constant 256 : index
    scf.for %i = %i0 to %iN step %i1 {
      %v = memref.load %in[%i] : memref<256xi8>
      %r = gf.inv %v : i8
      memref.store %r, %out[%i] : memref<256xi8>
    }
    return
  }

  // Bench: repeat N times (rolled)
  // Returns i8 accumulator; swaps %in/%out each iter (ping-pong)
func.func @bench_inv_domain(%iters: index,
                            %in: memref<256xi8>, %out: memref<256xi8>) -> i8 {
  %c0    = arith.constant 0   : index
  %c1    = arith.constant 1   : index
  %i0    = arith.constant 0   : index
  %i1    = arith.constant 1   : index
  %i256  = arith.constant 256 : index
  %acc0  = arith.constant 0   : i8

  // iter_args carry: (accumulator, current-in, current-out)
  %acc, %in_final, %out_final =
  scf.for %t = %c0 to %iters step %c1
      iter_args(%a = %acc0, %in_cur = %in, %out_cur = %out)
      -> (i8, memref<256xi8>, memref<256xi8>) {
    // 1) out_cur = inv(in_cur)
    func.call @inv_domain_kernel(%in_cur, %out_cur)
      : (memref<256xi8>, memref<256xi8>) -> ()

    // 2) per-iter consume: xor-reduce out_cur into running accumulator
    %a_next = scf.for %j = %i0 to %i256 step %i1 iter_args(%cur = %a) -> i8 {
      %v  = memref.load %out_cur[%j] : memref<256xi8>
      %nx = arith.xori %cur, %v : i8
      scf.yield %nx : i8
    }

    // 3) ping-pong: swap roles of in/out for next iteration
    scf.yield %a_next, %out_cur, %in_cur : i8, memref<256xi8>, memref<256xi8>
  }

  // We only need the accumulator; final buffers are there if you care about parity.
  return %acc : i8
  }



  // Main: init in[i] = i; run; XOR-reduce out to keep stores live
  func.func @main() -> i8 {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %iN = arith.constant 256 : index
    %iters = arith.constant 1000 : index

    %IN  = memref.alloca() : memref<256xi8>
    %OUT = memref.alloca() : memref<256xi8>

    scf.for %i = %i0 to %iN step %i1 {
      %ii = arith.index_cast %i : index to i8
      memref.store %ii, %IN[%i] : memref<256xi8>
    }

    %acc = func.call @bench_inv_domain(%iters, %IN, %OUT)
         : (index, memref<256xi8>, memref<256xi8>) -> i8
    return %acc : i8

  }
}
