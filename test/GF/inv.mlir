module {
  // An example function using gf.inv with a non-zero argument.
  func.func @main() -> i8 {
    %a = arith.constant 5 : i8
    %c = gf.inv %a : i8
    func.return %c : i8
  }
}
