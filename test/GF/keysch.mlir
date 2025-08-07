module {
  func.func @main() -> (i8) {
    // 1) Allocate memory for key and schedule
    %key = memref.alloca() : memref<16xi8>
    %schedule = memref.alloca() : memref<176xi8>

    // 2) AES-128 test key (FIPS-197 Appendix A)
    %k0 = arith.constant 0x2b : i8
    %k1 = arith.constant 0x7e : i8
    %k2 = arith.constant 0x15 : i8
    %k3 = arith.constant 0x16 : i8
    %k4 = arith.constant 0x28 : i8
    %k5 = arith.constant 0xae : i8
    %k6 = arith.constant 0xd2 : i8
    %k7 = arith.constant 0xa6 : i8
    %k8 = arith.constant 0xab : i8
    %k9 = arith.constant 0xf7 : i8
    %k10 = arith.constant 0x15 : i8
    %k11 = arith.constant 0x88 : i8
    %k12 = arith.constant 0x09 : i8
    %k13 = arith.constant 0xcf : i8
    %k14 = arith.constant 0x4f : i8
    %k15 = arith.constant 0x3c : i8

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

    memref.store %k0, %key[%i0] : memref<16xi8>
    memref.store %k1, %key[%i1] : memref<16xi8>
    memref.store %k2, %key[%i2] : memref<16xi8>
    memref.store %k3, %key[%i3] : memref<16xi8>
    memref.store %k4, %key[%i4] : memref<16xi8>
    memref.store %k5, %key[%i5] : memref<16xi8>
    memref.store %k6, %key[%i6] : memref<16xi8>
    memref.store %k7, %key[%i7] : memref<16xi8>
    memref.store %k8, %key[%i8] : memref<16xi8>
    memref.store %k9, %key[%i9] : memref<16xi8>
    memref.store %k10, %key[%i10] : memref<16xi8>
    memref.store %k11, %key[%i11] : memref<16xi8>
    memref.store %k12, %key[%i12] : memref<16xi8>
    memref.store %k13, %key[%i13] : memref<16xi8>
    memref.store %k14, %key[%i14] : memref<16xi8>
    memref.store %k15, %key[%i15] : memref<16xi8>

    // 3) Run key schedule
    gf.key_schedule %key into %schedule
      : memref<16xi8>, memref<176xi8>

    // 4) Reduce result by XOR-ing all 176 bytes of the schedule
    %r176 = arith.constant 176 : index
    %zero = arith.constant 0 : i8
    %final = scf.for %j = %i0 to %r176 step %i1 iter_args(%acc = %zero) -> i8 {
      %val = memref.load %schedule[%j] : memref<176xi8>
      %new_acc = arith.xori %acc, %val : i8
      scf.yield %new_acc : i8
    }

    func.return %final : i8
  }
}
