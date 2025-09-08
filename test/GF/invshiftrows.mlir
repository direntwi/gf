module {
    func.func @main() -> (i8) {
    // 1) Allocate memory for state
    %state = memref.alloca() : memref<16xi8>

    // 2) AES-128 test key (FIPS-197 Appendix A)
    %k0 = arith.constant 0x7a : i8
    %k1 = arith.constant 0xd5 : i8
    %k2 = arith.constant 0xfd : i8
    %k3 = arith.constant 0xa7 : i8
    %k4 = arith.constant 0x89 : i8
    %k5 = arith.constant 0xef : i8
    %k6 = arith.constant 0x4e : i8
    %k7 = arith.constant 0x27 : i8
    %k8 = arith.constant 0x2b : i8
    %k9 = arith.constant 0xca : i8
    %k10 = arith.constant 0x10 : i8
    %k11 = arith.constant 0x0b : i8
    %k12 = arith.constant 0x3d : i8
    %k13 = arith.constant 0x9f : i8
    %k14 = arith.constant 0xf5 : i8
    %k15 = arith.constant 0x9f : i8

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

    memref.store %k0, %state[%i0] : memref<16xi8>
    memref.store %k1, %state[%i1] : memref<16xi8>
    memref.store %k2, %state[%i2] : memref<16xi8>
    memref.store %k3, %state[%i3] : memref<16xi8>
    memref.store %k4, %state[%i4] : memref<16xi8>
    memref.store %k5, %state[%i5] : memref<16xi8>
    memref.store %k6, %state[%i6] : memref<16xi8>
    memref.store %k7, %state[%i7] : memref<16xi8>
    memref.store %k8, %state[%i8] : memref<16xi8>
    memref.store %k9, %state[%i9] : memref<16xi8>
    memref.store %k10, %state[%i10] : memref<16xi8>
    memref.store %k11, %state[%i11] : memref<16xi8>
    memref.store %k12, %state[%i12] : memref<16xi8>
    memref.store %k13, %state[%i13] : memref<16xi8>
    memref.store %k14, %state[%i14] : memref<16xi8>
    memref.store %k15, %state[%i15] : memref<16xi8>

    // 3) Run key schedule
    gf.inv_shift_rows %state : memref<16xi8>

    // 4) Reduce result by XOR-ing all 16 bytes of the schedule
    // %r16 = arith.constant 16 : index
    // %zero = arith.constant 0 : i8
    // %final = scf.for %j = %i0 to %r16 step %i1 iter_args(%acc = %zero) -> i8 {
    //   %val = memref.load %state[%j] : memref<16xi8>
    //   %new_acc = arith.xori %acc, %val : i8
    //   scf.yield %new_acc : i8
    // }

    // func.return %final : i8


    %c01 = arith.constant 0 : index
    %iters = arith.constant 10001 : index

    %i16  = arith.constant 16 : index
    %acc0 = arith.constant 0  : i8

    // rolled outer loop with per-iter consume; returns the accumulator
    %acc = scf.for %t = %c01 to %iters step %i1 iter_args(%a = %acc0) -> i8 {
      // one ShiftRows (in-place)
      gf.inv_shift_rows %state : memref<16xi8>

      // per-iter XOR-reduce of the current state into running acc
      %a_next = scf.for %j = %i0 to %i16 step %i1 iter_args(%cur = %a) -> i8 {
        %v  = memref.load %state[%j] : memref<16xi8>
        %nx = arith.xori %cur, %v : i8
        scf.yield %nx : i8
      }

      scf.yield %a_next : i8
    }
    func.return %acc : i8
    
    }
}
