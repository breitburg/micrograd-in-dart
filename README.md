# micrograd _in Dart_

Implements backpropagation (reverse-mode autodiff) over a dynamically built directed acyclic graph (DAG) and a small neural networks library on top of it, with a PyTorch-like API. This project is inspired by [micrograd by Andrew Karpathy](https://github.com/karpathy/micrograd).

> [!IMPORTANT]
> This code is 100% organic. No AI was used in the process of writing this code.

## Usage

Find gradients of a simple expression:

```dart
final a = Value(3);
final b = Value(5);
final c = Value(2);

final f = a * b + c;
f.backward();

print(a.grad); // 5.0
print(b.grad); // 3.0
```

Or, train a simple multi-layer perceptron (MLP) to learn the logical OR function:

```dart
final mlp = MultiLayerPerceptron(
  neuronInputsCount: 2,
  layerNeuronCounts: [4, 4, 1],
);

final dataset = [
  ([0, 1], 1),
  ([0, 0], 0),
  ([1, 0], 1),
  ([1, 1], 1),
];

final learningStep = 5e-2;

for (var step = 0; step < 200; step++) {
  final losses = [
    for (final (inputs, target) in dataset)
      (mlp.forward(inputs).first - target).power(2),
  ];
  final loss = losses.reduce((value, element) => value + element);
  print('loss: ${loss.data}');

  mlp.zeroGrad();
  loss.backward();

  for (final parameter in mlp.parameters) {
    parameter.data -= parameter.grad * learningStep;
  }
}

final input = [0, 0];
print('Inference from $input is ${mlp.forward(input).first.data}'); // Should be close to 0
```

