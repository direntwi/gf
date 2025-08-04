module {
  func.func @main() -> i8 {
    %c30 = arith.constant 30 : i8
    %c10 = arith.constant 10 : i8
    %result = gf.add %c30, %c10 : i8
    func.return %result : i8
  }
}