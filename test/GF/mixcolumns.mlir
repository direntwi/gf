//For correctness:

// module {
//   memref.global "public" @_dump_start : memref<16xi8>
//   func.func @main() -> (i8) {
//     // 1) Allocate memory for input and output states
//     %state = memref.alloca() : memref<16xi8>
//     %out   = memref.alloca() : memref<16xi8>

//     // 2) Constants for the state (row-major 4x4)
//     %c0  = arith.constant 0xd4 : i8
//     %c1  = arith.constant 0xbf : i8
//     %c2  = arith.constant 0x5d : i8
//     %c3  = arith.constant 0x30 : i8
//     %c4  = arith.constant 0xe0 : i8
//     %c5  = arith.constant 0xb4 : i8
//     %c6  = arith.constant 0x52 : i8
//     %c7  = arith.constant 0xae : i8
//     %c8  = arith.constant 0xb8 : i8
//     %c9  = arith.constant 0x41 : i8
//     %c10 = arith.constant 0x11 : i8
//     %c11 = arith.constant 0xf1 : i8
//     %c12 = arith.constant 0x1e : i8
//     %c13 = arith.constant 0x27 : i8
//     %c14 = arith.constant 0x98 : i8
//     %c15 = arith.constant 0xe5 : i8

//     // 3) Store constants into state
//     %i0  = arith.constant 0  : index
//     %i1  = arith.constant 1  : index
//     %i2  = arith.constant 2  : index
//     %i3  = arith.constant 3  : index
//     %i4  = arith.constant 4  : index
//     %i5  = arith.constant 5  : index
//     %i6  = arith.constant 6  : index
//     %i7  = arith.constant 7  : index
//     %i8  = arith.constant 8  : index
//     %i9  = arith.constant 9  : index
//     %i10 = arith.constant 10 : index
//     %i11 = arith.constant 11 : index
//     %i12 = arith.constant 12 : index
//     %i13 = arith.constant 13 : index
//     %i14 = arith.constant 14 : index
//     %i15 = arith.constant 15 : index

//     memref.store %c0,  %state[%i0]  : memref<16xi8>
//     memref.store %c1,  %state[%i1]  : memref<16xi8>
//     memref.store %c2,  %state[%i2]  : memref<16xi8>
//     memref.store %c3,  %state[%i3]  : memref<16xi8>
//     memref.store %c4,  %state[%i4]  : memref<16xi8>
//     memref.store %c5,  %state[%i5]  : memref<16xi8>
//     memref.store %c6,  %state[%i6]  : memref<16xi8>
//     memref.store %c7,  %state[%i7]  : memref<16xi8>
//     memref.store %c8,  %state[%i8]  : memref<16xi8>
//     memref.store %c9,  %state[%i9]  : memref<16xi8>
//     memref.store %c10, %state[%i10] : memref<16xi8>
//     memref.store %c11, %state[%i11] : memref<16xi8>
//     memref.store %c12, %state[%i12] : memref<16xi8>
//     memref.store %c13, %state[%i13] : memref<16xi8>
//     memref.store %c14, %state[%i14] : memref<16xi8>
//     memref.store %c15, %state[%i15] : memref<16xi8>

//     // 4) Loop through the 4 columns (each column has 4 rows)
//     %c4_idx = arith.constant 4 : index
//     scf.for %col = %i0 to %c4_idx step %i1 {
//       %col_mem = memref.alloca() : memref<4xi8>
//       %out_col = memref.alloca() : memref<4xi8>

//       %base = arith.muli %col, %c4_idx : index
//       %row0 = arith.addi %base, %i0 : index
//       %row1 = arith.addi %base, %i1 : index
//       %row2 = arith.addi %base, %i2 : index
//       %row3 = arith.addi %base, %i3 : index

//       %v0 = memref.load %state[%row0] : memref<16xi8>
//       %v1 = memref.load %state[%row1] : memref<16xi8>
//       %v2 = memref.load %state[%row2] : memref<16xi8>
//       %v3 = memref.load %state[%row3] : memref<16xi8>

//       memref.store %v0, %col_mem[%i0] : memref<4xi8>
//       memref.store %v1, %col_mem[%i1] : memref<4xi8>
//       memref.store %v2, %col_mem[%i2] : memref<4xi8>
//       memref.store %v3, %col_mem[%i3] : memref<4xi8>

//       gf.mix_columns %col_mem into %out_col
//         : memref<4xi8>, memref<4xi8>

//       %o0 = memref.load %out_col[%i0] : memref<4xi8>
//       %o1 = memref.load %out_col[%i1] : memref<4xi8>
//       %o2 = memref.load %out_col[%i2] : memref<4xi8>
//       %o3 = memref.load %out_col[%i3] : memref<4xi8>

//       memref.store %o0, %out[%row0] : memref<16xi8>
//       memref.store %o1, %out[%row1] : memref<16xi8>
//       memref.store %o2, %out[%row2] : memref<16xi8>
//       memref.store %o3, %out[%row3] : memref<16xi8>

//     }

//     // 5) XOR reduction using loop-carried value
//     %r16 = arith.constant 16 : index
//     %zero = arith.constant 0 : i8
//     %final = scf.for %j = %i0 to %r16 step %i1 iter_args(%acc = %zero) -> i8 {
//       %val = memref.load %out[%j] : memref<16xi8>
//       %new_acc = arith.xori %acc, %val : i8
//       scf.yield %new_acc : i8
//     }

//     // After XOR loop
//     %dump = memref.get_global @_dump_start : memref<16xi8>
//     scf.for %k = %i0 to %r16 step %i1 {
//       %v = memref.load %out[%k] : memref<16xi8>
//       memref.store %v, %dump[%k] : memref<16xi8>
//     }

//     func.return %final : i8
//   }
// }

// For benchmarking:
module {
  // --- Kernel: one MixColumns over a 16-byte state ---
  func.func @mixcolumns_kernel(%state: memref<16xi8>, %out: memref<16xi8>) {
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i2  = arith.constant 2  : index
    %i3  = arith.constant 3  : index
    %i4  = arith.constant 4  : index
    // Reuse small scratch buffers (no per-iteration alloca)
    %col_mem = memref.alloca() : memref<4xi8>
    %out_col = memref.alloca() : memref<4xi8>

    scf.for %col = %i0 to %i4 step %i1 {
      %base = arith.muli %col, %i4 : index
      %row0 = arith.addi %base, %i0 : index
      %row1 = arith.addi %base, %i1 : index
      %row2 = arith.addi %base, %i2 : index
      %row3 = arith.addi %base, %i3 : index

      %v0 = memref.load %state[%row0] : memref<16xi8>
      %v1 = memref.load %state[%row1] : memref<16xi8>
      %v2 = memref.load %state[%row2] : memref<16xi8>
      %v3 = memref.load %state[%row3] : memref<16xi8>

      memref.store %v0, %col_mem[%i0] : memref<4xi8>
      memref.store %v1, %col_mem[%i1] : memref<4xi8>
      memref.store %v2, %col_mem[%i2] : memref<4xi8>
      memref.store %v3, %col_mem[%i3] : memref<4xi8>

      gf.mix_columns %col_mem into %out_col : memref<4xi8>, memref<4xi8>

      %o0 = memref.load %out_col[%i0] : memref<4xi8>
      %o1 = memref.load %out_col[%i1] : memref<4xi8>
      %o2 = memref.load %out_col[%i2] : memref<4xi8>
      %o3 = memref.load %out_col[%i3] : memref<4xi8>

      memref.store %o0, %out[%row0] : memref<16xi8>
      memref.store %o1, %out[%row1] : memref<16xi8>
      memref.store %o2, %out[%row2] : memref<16xi8>
      memref.store %o3, %out[%row3] : memref<16xi8>
    }
    return
  }

  // --- Harness: repeat kernel N times (rolled outer loop) ---
  func.func @bench_mixcolumns(%iters: index, %state: memref<16xi8>, %out: memref<16xi8>) -> i8 {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %i0   = arith.constant 0 : index
  %i1   = arith.constant 1 : index
  %i16  = arith.constant 16 : index
  %acc0 = arith.constant 0 : i8

  %acc = scf.for %t = %c0 to %iters step %c1 iter_args(%a = %acc0) -> i8 {
    // 1) do one MixColumns
    func.call @mixcolumns_kernel(%state, %out) : (memref<16xi8>, memref<16xi8>) -> ()

    // 2) loop-carried dep: state <- out  (prevents loop collapse/hoist)
    scf.for %i = %i0 to %i16 step %i1 {
      %v = memref.load %out[%i] : memref<16xi8>
      memref.store %v, %state[%i] : memref<16xi8>
    }

    // 3) per-iter consume: XOR-reduce out into running accumulator
    %a_next = scf.for %j = %i0 to %i16 step %i1 iter_args(%cur = %a) -> i8 {
      %v  = memref.load %out[%j] : memref<16xi8>
      %nx = arith.xori %cur, %v : i8
      scf.yield %nx : i8
    }
    scf.yield %a_next : i8
  }
  return %acc : i8
}


  // --- Main: init once, then run 1000 iterations ---
  func.func @main() -> i8 {
    %state = memref.alloca() : memref<16xi8>
    %out   = memref.alloca() : memref<16xi8>


    %iters = arith.constant 1000 : index
  %acc   = func.call @bench_mixcolumns(%iters, %state, %out)
            : (index, memref<16xi8>, memref<16xi8>) -> i8
  return %acc : i8


  }
}
