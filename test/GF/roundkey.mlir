module {
    func.func @main() -> (i8) {
        // 1) Allocate memory for input and output states
        %state = memref.alloca() : memref<16xi8>
        %key   = memref.alloca() : memref<16xi8>
    
        // 2) FIPS 197 Round 0 state
        %s0 = arith.constant 0x32 : i8
        %s1 = arith.constant 0x43 : i8
        %s2 = arith.constant 0xf6 : i8
        %s3 = arith.constant 0xa8 : i8
        %s4 = arith.constant 0x88 : i8
        %s5 = arith.constant 0x5a : i8
        %s6 = arith.constant 0x30 : i8
        %s7 = arith.constant 0x8d : i8
        %s8 = arith.constant 0x31 : i8
        %s9 = arith.constant 0x31 : i8
        %s10 = arith.constant 0x98 : i8
        %s11 = arith.constant 0xa2 : i8
        %s12 = arith.constant 0xe0 : i8
        %s13 = arith.constant 0x37 : i8
        %s14 = arith.constant 0x07 : i8
        %s15 = arith.constant 0x34 : i8

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
    
        // 3) Indices
        %i0  = arith.constant 0  : index
        %i1  = arith.constant 1  : index
        %i2  = arith.constant 2  : index
        %i3  = arith.constant 3  : index
        %i4  = arith.constant 4  : index
        %i5  = arith.constant 5  : index
        %i6  = arith.constant 6  : index
        %i7  = arith.constant 7  : index
        %i8  = arith.constant 8  : index
        %i9  = arith.constant 9  : index
        %i10 = arith.constant 10 : index
        %i11 = arith.constant 11 : index
        %i12 = arith.constant 12 : index
        %i13 = arith.constant 13 : index
        %i14 = arith.constant 14 : index
        %i15 = arith.constant 15 : index

        // 5) Store state
        memref.store %s0, %state[%i0] : memref<16xi8>
        memref.store %s1, %state[%i1] : memref<16xi8>
        memref.store %s2, %state[%i2] : memref<16xi8>
        memref.store %s3, %state[%i3] : memref<16xi8>
        memref.store %s4, %state[%i4] : memref<16xi8>
        memref.store %s5, %state[%i5] : memref<16xi8>
        memref.store %s6, %state[%i6] : memref<16xi8>
        memref.store %s7, %state[%i7] : memref<16xi8>
        memref.store %s8, %state[%i8] : memref<16xi8>
        memref.store %s9, %state[%i9] : memref<16xi8>
        memref.store %s10, %state[%i10] : memref<16xi8>
        memref.store %s11, %state[%i11] : memref<16xi8>
        memref.store %s12, %state[%i12] : memref<16xi8>
        memref.store %s13, %state[%i13] : memref<16xi8>
        memref.store %s14, %state[%i14] : memref<16xi8>
        memref.store %s15, %state[%i15] : memref<16xi8>

        // 6) Store round key
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

        // // 7) Apply AddRoundKey operation
        // gf.add_round_key %state, %key : memref<16xi8>, memref<16xi8>

        // // 8) Verify by XOR-ing entire result
        // %n = arith.constant 16 : index
        // %zero = arith.constant 0 : i8
        // %final = scf.for %j = %i0 to %n step %i1 iter_args(%acc = %zero) -> i8 {
        // %val = memref.load %state[%j] : memref<16xi8>
        // %new = arith.xori %acc, %val : i8
        // scf.yield %new : i8
        // }

        // func.return %final : i8

        %c01 = arith.constant 0 : index
        %iters = arith.constant 10001 : index
        %c16  = arith.constant 16 : index

        %acc0 = arith.constant 0 : i8
        %acc  = scf.for %t = %c01 to %iters step %i1 iter_args(%a = %acc0) -> i8 {
        // one AddRoundKey (in-place on %state)
        gf.add_round_key %state, %key : memref<16xi8>, memref<16xi8>

        // per-iter consume: XOR-reduce current state into running acc
        %a_next = scf.for %j = %i0 to %c16 step %i1 iter_args(%cur = %a) -> i8 {
            %v  = memref.load %state[%j] : memref<16xi8>
            %nx = arith.xori %cur, %v : i8
            scf.yield %nx : i8
        }
        scf.yield %a_next : i8
        }

        return %acc : i8


    } 
}