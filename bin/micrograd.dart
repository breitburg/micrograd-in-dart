import 'package:micrograd/micrograd.dart';

void main(Iterable<String> arguments) async {
  final a = Value(3);
  final b = Value(5);
  final c = Value(2);

  final f = a * b + c;
  f.backward();

  print(a.grad); // 5.0
  print(b.grad); // 3.0
}
