// module {
//   func.func @main() -> i8 {
//     %c30 = arith.constant 30 : i8
//     %c10 = arith.constant 10 : i8
//     %result = gf.add %c30, %c10 : i8
//     func.return %result : i8
//   }
// }

// For benchmarking

module {
  // Kernel: C[i] = A[i] (+) B[i] over GF(2^8) for 16 bytes
  func.func @add_block16_kernel(%a: memref<16xi8>, %b: memref<16xi8>, %c: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i16 = arith.constant 16 : index
    scf.for %i = %i0 to %i16 step %i1 {
      %av = memref.load %a[%i] : memref<16xi8>
      %bv = memref.load %b[%i] : memref<16xi8>
      %sv = gf.add %av, %bv : i8
      memref.store %sv, %c[%i] : memref<16xi8>
    }
    return
  }

  // Bench: repeat N times (rolled)
  // returns i8 accumulator
  func.func @bench_add_block16(%iters: index,
                             %a: memref<16xi8>, %b: memref<16xi8>, %c: memref<16xi8>) -> i8 {
    %z    = arith.constant 0  : index
    %one  = arith.constant 1  : index
    %i0   = arith.constant 0  : index
    %i1   = arith.constant 1  : index
    %i16  = arith.constant 16 : index
    %acc0 = arith.constant 0  : i8

    %acc = scf.for %t = %z to %iters step %one iter_args(%aacc = %acc0) -> i8 {
      func.call @add_block16_kernel(%a, %b, %c)
        : (memref<16xi8>, memref<16xi8>, memref<16xi8>) -> ()

      // loop-carried dependency: A <- C  (prevents loop collapse)
      scf.for %i = %i0 to %i16 step %i1 {
        %w = memref.load %c[%i] : memref<16xi8>
        memref.store %w, %a[%i] : memref<16xi8>
      }

      // per-iter consume: xor C into running acc
      %aacc_next = scf.for %j = %i0 to %i16 step %i1 iter_args(%cur = %aacc) -> i8 {
        %v  = memref.load %c[%j] : memref<16xi8>
        %nx = arith.xori %cur, %v : i8
        scf.yield %nx : i8
      }
      scf.yield %aacc_next : i8
    }
    return %acc : i8
  }



  // Main: init A,B once; run; XOR-reduce C to keep stores live
  func.func @main() -> i8 {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i16 = arith.constant 16 : index
    %iters = arith.constant 10000 : index

    %A = memref.alloca() : memref<16xi8>
    %B = memref.alloca() : memref<16xi8>
    %C = memref.alloca() : memref<16xi8>

    
    scf.for %i = %i0 to %i16 step %i1 {
      %ii = arith.index_cast %i : index to i8
      memref.store %ii, %A[%i] : memref<16xi8>
      %i8_3 = arith.constant 3 : i8
      %i8_1 = arith.constant 1 : i8
      %t0   = arith.muli %ii, %i8_3 : i8
      %t1   = arith.addi %t0, %i8_1 : i8
      memref.store %t1, %B[%i] : memref<16xi8>

    }

    %acc = func.call @bench_add_block16(%iters, %A, %B, %C)
      : (index, memref<16xi8>, memref<16xi8>, memref<16xi8>) -> i8
    return %acc : i8


   
  }
}
