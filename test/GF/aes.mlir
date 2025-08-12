module {
    memref.global "public" @_dump_start : memref<16xi8>

    func.func @aes_encrypt_driver(%key_in: memref<16xi8>, %state_in_out: memref<16xi8>) -> () {
        
        %schedule = memref.alloca() : memref<176xi8>            // 11 * 16

        // Expand key schedule
        gf.key_schedule %key_in into %schedule : memref<16xi8>, memref<176xi8>

        // Initial AddRoundKey (round 0)
        %rk0 = memref.alloca() : memref<16xi8>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %c16 step %c1 {
            %v = memref.load %schedule[%i] : memref<176xi8>
            memref.store %v, %rk0[%i] : memref<16xi8>
            scf.yield
        }
        gf.add_round_key %state_in_out, %rk0 : memref<16xi8>, memref<16xi8>

        // Buffers for MixColumns
        %col_in = memref.alloca() : memref<4xi8>
        %col_out = memref.alloca() : memref<4xi8>

        // Column offsets (col-major)
        %c_col0 = arith.constant 0 : index
        %c_col1 = arith.constant 4 : index
        %c_col2 = arith.constant 8 : index
        %c_col3 = arith.constant 12 : index

        // Rounds 1..9
        %r1 = arith.constant 1 : index
        %r10 = arith.constant 10 : index
        scf.for %round = %r1 to %r10 step %c1 {
            // SubBytes
            scf.for %i = %c0 to %c16 step %c1 {
            %b = memref.load %state_in_out[%i] : memref<16xi8>
            %b_s = gf.sbox %b : i8
            memref.store %b_s, %state_in_out[%i] : memref<16xi8>
            scf.yield
            }

            // ShiftRows
            gf.shift_rows %state_in_out : memref<16xi8>

            // MixColumns
            %r4 = arith.constant 4 : index
            scf.for %j = %c0 to %c4 step %c1 {
                %base = arith.muli %j, %r4 : index

                %p0 = arith.addi %base, %c0 : index
                %p1 = arith.addi %base, %c1 : index
                %p2 = arith.addi %base, %c2 : index
                %p3 = arith.addi %base, %c3 : index

                %v0 = memref.load %state_in_out[%p0] : memref<16xi8>
                %v1 = memref.load %state_in_out[%p1] : memref<16xi8>
                %v2 = memref.load %state_in_out[%p2] : memref<16xi8>
                %v3 = memref.load %state_in_out[%p3] : memref<16xi8>

                memref.store %v0, %col_in[%c0] : memref<4xi8>
                memref.store %v1, %col_in[%c1] : memref<4xi8>
                memref.store %v2, %col_in[%c2] : memref<4xi8>
                memref.store %v3, %col_in[%c3] : memref<4xi8>

                gf.mix_columns %col_in into %col_out : memref<4xi8>, memref<4xi8>

                %rl0 = memref.load %col_out[%c0] : memref<4xi8>
                %rl1 = memref.load %col_out[%c1] : memref<4xi8>
                %rl2 = memref.load %col_out[%c2] : memref<4xi8>
                %rl3 = memref.load %col_out[%c3] : memref<4xi8>

                memref.store %rl0, %state_in_out[%p0] : memref<16xi8>
                memref.store %rl1, %state_in_out[%p1] : memref<16xi8>
                memref.store %rl2, %state_in_out[%p2] : memref<16xi8>
                memref.store %rl3, %state_in_out[%p3] : memref<16xi8>

                scf.yield
            }

            // AddRoundKey for this round
            %off_mul = arith.constant 16 : index
            %off = arith.muli %round, %off_mul : index
            scf.for %jj = %c0 to %c16 step %c1 {
                %src_idx = arith.addi %off, %jj : index
                %val = memref.load %schedule[%src_idx] : memref<176xi8>
                memref.store %val, %rk0[%jj] : memref<16xi8>
                scf.yield
            }
            gf.add_round_key %state_in_out, %rk0 : memref<16xi8>, memref<16xi8>
            scf.yield
        }

        // Final round (round 10)
        scf.for %i = %c0 to %c16 step %c1 {
            %b = memref.load %state_in_out[%i] : memref<16xi8>
            %b_s = gf.sbox %b : i8
            memref.store %b_s, %state_in_out[%i] : memref<16xi8>
            scf.yield
        }
        gf.shift_rows %state_in_out : memref<16xi8>

        %off10 = arith.constant 160 : index
        scf.for %jj = %c0 to %c16 step %c1 {
            %src_idx10 = arith.addi %off10, %jj : index
            %val10 = memref.load %schedule[%src_idx10] : memref<176xi8>
            memref.store %val10, %rk0[%jj] : memref<16xi8>
            scf.yield
        }
        gf.add_round_key %state_in_out, %rk0 : memref<16xi8>, memref<16xi8>

        // %final = arith.constant 1 : i8
        func.return
    }

    func.func @main() -> i8 {
        // allocate key and state on the stack
        %key = memref.alloca() : memref<16xi8>
        %state = memref.alloca() : memref<16xi8>

        // constants we will store (i8 constants)
        %idx0 = arith.constant 0 : index
        %idx1 = arith.constant 1 : index
        %idx2 = arith.constant 2 : index
        %idx3 = arith.constant 3 : index
        %idx4 = arith.constant 4 : index
        %idx5 = arith.constant 5 : index
        %idx6 = arith.constant 6 : index
        %idx7 = arith.constant 7 : index
        %idx8 = arith.constant 8 : index
        %idx9 = arith.constant 9 : index
        %idx10 = arith.constant 10 : index
        %idx11 = arith.constant 11 : index
        %idx12 = arith.constant 12 : index
        %idx13 = arith.constant 13 : index
        %idx14 = arith.constant 14 : index
        %idx15 = arith.constant 15 : index

        // Known key for AES-128
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

        // store key bytes 0..15 into %key
        memref.store %k0, %key[%idx0] : memref<16xi8>
        memref.store %k1, %key[%idx1] : memref<16xi8>
        memref.store %k2, %key[%idx2] : memref<16xi8>
        memref.store %k3, %key[%idx3] : memref<16xi8>
        memref.store %k4, %key[%idx4] : memref<16xi8>
        memref.store %k5, %key[%idx5] : memref<16xi8>
        memref.store %k6, %key[%idx6] : memref<16xi8>
        memref.store %k7, %key[%idx7] : memref<16xi8>
        memref.store %k8, %key[%idx8] : memref<16xi8>
        memref.store %k9, %key[%idx9] : memref<16xi8>
        memref.store %k10, %key[%idx10] : memref<16xi8>
        memref.store %k11, %key[%idx11] : memref<16xi8>
        memref.store %k12, %key[%idx12] : memref<16xi8>
        memref.store %k13, %key[%idx13] : memref<16xi8>
        memref.store %k14, %key[%idx14] : memref<16xi8>
        memref.store %k15, %key[%idx15] : memref<16xi8>

        // store plaintext bytes into state (plaintext: 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff)
        %pt0 = arith.constant 0x32 : i8
        %pt1 = arith.constant 0x43 : i8
        %pt2 = arith.constant 0xf6 : i8
        %pt3 = arith.constant 0xa8 : i8
        %pt4 = arith.constant 0x88 : i8
        %pt5 = arith.constant 0x5a : i8
        %pt6 = arith.constant 0x30 : i8
        %pt7 = arith.constant 0x8d : i8
        %pt8 = arith.constant 0x31 : i8
        %pt9 = arith.constant 0x31 : i8
        %pt10 = arith.constant 0x98 : i8
        %pt11 = arith.constant 0xa2 : i8
        %pt12 = arith.constant 0xe0 : i8
        %pt13 = arith.constant 0x37 : i8
        %pt14 = arith.constant 0x07 : i8
        %pt15 = arith.constant 0x34 : i8

        // store plaintext bytes into state positions 0..15
        memref.store %pt0, %state[%idx0] : memref<16xi8>
        memref.store %pt1, %state[%idx1] : memref<16xi8>
        memref.store %pt2, %state[%idx2] : memref<16xi8>
        memref.store %pt3, %state[%idx3] : memref<16xi8>
        memref.store %pt4, %state[%idx4] : memref<16xi8>
        memref.store %pt5, %state[%idx5] : memref<16xi8>
        memref.store %pt6, %state[%idx6] : memref<16xi8>
        memref.store %pt7, %state[%idx7] : memref<16xi8>
        memref.store %pt8, %state[%idx8] : memref<16xi8>
        memref.store %pt9, %state[%idx9] : memref<16xi8>
        memref.store %pt10, %state[%idx10] : memref<16xi8>
        memref.store %pt11, %state[%idx11] : memref<16xi8>
        memref.store %pt12, %state[%idx12] : memref<16xi8>
        memref.store %pt13, %state[%idx13] : memref<16xi8>
        memref.store %pt14, %state[%idx14] : memref<16xi8>
        memref.store %pt15, %state[%idx15] : memref<16xi8>

        // call the driver (in-place on %state)
        func.call @aes_encrypt_driver(%key, %state) : (memref<16xi8>, memref<16xi8>) -> ()

        // 5) XOR reduction using loop-carried value
        %r16 = arith.constant 16 : index
        %zero = arith.constant 0 : i8
        %final = scf.for %j = %idx0 to %r16 step %idx1 iter_args(%acc = %zero) -> i8 {
        %val = memref.load %state[%j] : memref<16xi8>
        %new_acc = arith.xori %acc, %val : i8
        scf.yield %new_acc : i8
        }

        // After XOR loop
        %dump = memref.get_global @_dump_start : memref<16xi8>
        scf.for %k = %idx0 to %r16 step %idx1 {
        %v = memref.load %state[%k] : memref<16xi8>
        memref.store %v, %dump[%k] : memref<16xi8>
        }

        func.return %final : i8

        
    }
}