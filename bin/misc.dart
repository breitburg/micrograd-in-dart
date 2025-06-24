import 'dart:convert';
import 'package:micrograd/engine.dart';

void trainingSimpleEquation() {
  final x = Value(3);
  final y = Value(4);

  // Trainable
  final a = Value(0);
  final b = Value(0);
  final c = Value(0);
  final target = 35;

  Value f;
  Value loss;

  final learningStep = 1e-6;
  while (true) {
    // Compute forward pass with current parameters
    f = a * x + b * y + c;
    loss = (f - target).power(2);

    // Compute gradients
    loss.backward();

    // Update parameters
    a.data -= a.grad * learningStep;
    b.data -= b.grad * learningStep;
    c.data -= c.grad * learningStep;

    // Zero gradients for next iteration
    a.grad = 0;
    b.grad = 0;
    c.grad = 0;

    print(
      'loss: ${loss.data}; a: ${a.data}; b: ${b.data}; c: ${c.data} = ${f.data}',
    );
  }
}

void neuron() {
  // inputs x1, x2
  final x1 = Value(2.0)..label = 'x1';
  final x2 = Value(0.0)..label = 'x2';

  // weights w1, w2
  final w1 = Value(-3.0)..label = 'w1';
  final w2 = Value(1.0)..label = 'w2';

  // bias b
  final b = Value(6.8813735878195437)..label = 'b';

  final x1w1 = x1 * w1;
  x1w1.label = 'x1*w1';
  final x2w2 = x2 * w2;
  x2w2.label = 'x2*w2';
  final x1w1x2w2 = x1w1 + x2w2;
  x1w1x2w2.label = 'x1w1+x2w2';
  final n = x1w1x2w2 + b;
  n.label = 'n';
  final o = n.tanh();

  o.backward();
  print(x1.grad);
}

void manualBackpropagation() {
  final a = Value(2.0)..label = 'a';
  a.grad = 6.0;
  final b = Value(-3.0)..label = 'b';
  b.grad = -4.0;
  final c = Value(10.0)..label = 'c';
  c.grad = -2.0;
  final e = a * b;
  e.label = 'e';
  e.grad = -2.0;
  final d = e + c;
  d.label = 'd';
  d.grad = -2.0;
  final f = Value(-2.0)..label = 'f';
  f.grad = 4.0;
  final L = d * f;
  L.label = 'L';
  L.grad = 1.0;
  print(jsonEncode(L.toJson()));
}

void partialDerivative() {
  final h = 0.00001;

  f(a, b, c) => a * b + c;
  final a = 2.0;
  final b = -3.0;
  final c = 10.0;

  final derivative = (f(a + h, b, c) - f(a, b, c)) / h;

  print(derivative);
}
