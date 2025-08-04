module {
  func.func @main() -> i8 {
    %zero = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index

    // Allocate A, B, C on the stack
    %A = memref.alloca() : memref<100xi8>
    %B = memref.alloca() : memref<100xi8>
    %C = memref.alloca() : memref<100xi8>

    %one = arith.constant 1 : i8
    scf.for %i = %c0 to %c100 step %c1 {
      %val = arith.index_cast %i : index to i8
      %val_plus_one = arith.addi %val, %one : i8
      memref.store %val_plus_one, %A[%i] : memref<100xi8>
    }

    %two = arith.constant 2 : i8
    scf.for %i = %c0 to %c100 step %c1 {
      %val = arith.index_cast %i : index to i8
      %val_plus_one = arith.addi %val, %one : i8
      %double = arith.muli %val_plus_one, %two : i8
      memref.store %double, %B[%i] : memref<100xi8>
    }


    // Zero initialize C[i]
    scf.for %i = %c0 to %c100 step %c1 {
      memref.store %zero, %C[%i] : memref<100xi8>
    }

    // Call gf.matmul
    gf.matmul
      [%A : memref<100xi8>]
      by [%B : memref<100xi8>]
      into %C
      {rowsA = 10 : i8, colsA = 10 : i8, colsB = 10 : i8}
      : memref<100xi8>

    // XOR accumulate all values in C
    %acc0 = arith.constant 0 : i8
    %result = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc = %acc0) -> i8 {
      %val = memref.load %C[%i] : memref<100xi8>
      %next = arith.xori %acc, %val : i8
      scf.yield %next : i8
    }

    return %result : i8
  }
}
