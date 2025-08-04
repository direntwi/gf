module {
  // An example function using galois.sbox
  func.func @main() -> i8 {
    %a = arith.constant 5 : i8
    %c = gf.sbox %a : i8
    func.return %c : i8 
  }
}
