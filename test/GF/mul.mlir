module {
  // An example function using gf.mul with non-zero arguments.
  func.func @main() -> i8 {
    %a = arith.constant 5 : i8
    %b = arith.constant 7 : i8
    %c = gf.mul %a, %b : i8
    func.return %c : i8
  }
}
