// For correctness

// module {
//   // An example function using galois.sbox
//   func.func @main() -> i8 {
//     %a = arith.constant 5 : i8
//     %c = gf.sbox %a : i8
//     func.return %c : i8 
//   }
// }


// For benchmarking
module {
  // --- Kernel: apply S-box to a 16-byte state in-place ---
  func.func @subbytes_kernel(%state: memref<16xi8>) {
    %i0 = arith.constant 0  : index
    %i1 = arith.constant 1  : index
    %i2 = arith.constant 2  : index
    %i3 = arith.constant 3  : index
    %i4 = arith.constant 4  : index
    %i5 = arith.constant 5  : index
    %i6 = arith.constant 6  : index
    %i7 = arith.constant 7  : index
    %i8 = arith.constant 8  : index
    %i9 = arith.constant 9  : index
    %i10 = arith.constant 10 : index
    %i11 = arith.constant 11 : index
    %i12 = arith.constant 12 : index
    %i13 = arith.constant 13 : index
    %i14 = arith.constant 14 : index
    %i15 = arith.constant 15 : index

    %b0  = memref.load %state[%i0]  : memref<16xi8>
    %b1  = memref.load %state[%i1]  : memref<16xi8>
    %b2  = memref.load %state[%i2]  : memref<16xi8>
    %b3  = memref.load %state[%i3]  : memref<16xi8>
    %b4  = memref.load %state[%i4]  : memref<16xi8>
    %b5  = memref.load %state[%i5]  : memref<16xi8>
    %b6  = memref.load %state[%i6]  : memref<16xi8>
    %b7  = memref.load %state[%i7]  : memref<16xi8>
    %b8  = memref.load %state[%i8]  : memref<16xi8>
    %b9  = memref.load %state[%i9]  : memref<16xi8>
    %b10 = memref.load %state[%i10] : memref<16xi8>
    %b11 = memref.load %state[%i11] : memref<16xi8>
    %b12 = memref.load %state[%i12] : memref<16xi8>
    %b13 = memref.load %state[%i13] : memref<16xi8>
    %b14 = memref.load %state[%i14] : memref<16xi8>
    %b15 = memref.load %state[%i15] : memref<16xi8>

    %s0  = gf.sbox %b0  : i8
    %s1  = gf.sbox %b1  : i8
    %s2  = gf.sbox %b2  : i8
    %s3  = gf.sbox %b3  : i8
    %s4  = gf.sbox %b4  : i8
    %s5  = gf.sbox %b5  : i8
    %s6  = gf.sbox %b6  : i8
    %s7  = gf.sbox %b7  : i8
    %s8  = gf.sbox %b8  : i8
    %s9  = gf.sbox %b9  : i8
    %s10 = gf.sbox %b10 : i8
    %s11 = gf.sbox %b11 : i8
    %s12 = gf.sbox %b12 : i8
    %s13 = gf.sbox %b13 : i8
    %s14 = gf.sbox %b14 : i8
    %s15 = gf.sbox %b15 : i8

    memref.store %s0,  %state[%i0]  : memref<16xi8>
    memref.store %s1,  %state[%i1]  : memref<16xi8>
    memref.store %s2,  %state[%i2]  : memref<16xi8>
    memref.store %s3,  %state[%i3]  : memref<16xi8>
    memref.store %s4,  %state[%i4]  : memref<16xi8>
    memref.store %s5,  %state[%i5]  : memref<16xi8>
    memref.store %s6,  %state[%i6]  : memref<16xi8>
    memref.store %s7,  %state[%i7]  : memref<16xi8>
    memref.store %s8,  %state[%i8]  : memref<16xi8>
    memref.store %s9,  %state[%i9]  : memref<16xi8>
    memref.store %s10, %state[%i10] : memref<16xi8>
    memref.store %s11, %state[%i11] : memref<16xi8>
    memref.store %s12, %state[%i12] : memref<16xi8>
    memref.store %s13, %state[%i13] : memref<16xi8>
    memref.store %s14, %state[%i14] : memref<16xi8>
    memref.store %s15, %state[%i15] : memref<16xi8>
    return
  }

  
  // --- Harness: repeat N times (rolled), return i8 accumulator ---
  func.func @bench_subbytes(%iters: index, %state: memref<16xi8>) -> i8 {
    %c0   = arith.constant 0  : index
    %c1   = arith.constant 1  : index
    %i0   = arith.constant 0  : index
    %i1   = arith.constant 1  : index
    %i16  = arith.constant 16 : index
    %acc0 = arith.constant 0  : i8

    %acc = scf.for %t = %c0 to %iters step %c1 iter_args(%a = %acc0) -> i8 {
      func.call @subbytes_kernel(%state) : (memref<16xi8>) -> ()
      // per-iter consume: XOR-reduce current state into running acc
      %a_next = scf.for %j = %i0 to %i16 step %i1 iter_args(%cur = %a) -> i8 {
        %v  = memref.load %state[%j] : memref<16xi8>
        %nx = arith.xori %cur, %v : i8
        scf.yield %nx : i8
      }
      scf.yield %a_next : i8
    }
    return %acc : i8
  }


  // --- Main: allocate once, run 5000 iterations ---
  func.func @main() -> i8 {
    %state = memref.alloca() : memref<16xi8>
    %iters = arith.constant 5000 : index
    %acc = func.call @bench_subbytes(%iters, %state)
            : (index, memref<16xi8>) -> i8
    return %acc : i8

  }
}
