import 'dart:math';

import 'package:micrograd/engine.dart';

final _random = Random();

abstract class Module {
  void zeroGrad() => [for (final parameter in parameters) parameter.grad = 0];
  List<Value> get parameters;
}

class Neuron extends Module {
  final int inputsCount;
  final Iterable<Value> weights;
  final Value bias;

  Neuron({required this.inputsCount})
    : weights = List.generate(
        inputsCount,
        (i) => Value(_random.nextDouble() * 2 - 1)..label = 'w$i',
      ),
      bias = Value(_random.nextDouble() * 2 - 1)..label = 'bias';

  Value forward(Iterable<Value> inputs) {
    final act =
        [
          for (var i = 0; i < inputsCount; i++)
            weights.elementAt(i) * inputs.elementAt(i),
        ].reduce((a, b) => a + b) +
        bias;

    return act.tanh();
  }

  @override
  List<Value> get parameters => [...weights, bias];
}

class Layer extends Module {
  // Input size
  final int neuronInputsCount;

  // Output size
  final int neuronsCount;
  final Iterable<Neuron> neurons;

  Layer({required this.neuronInputsCount, required this.neuronsCount})
    : neurons = List.generate(
        neuronsCount,
        (_) => Neuron(inputsCount: neuronInputsCount),
      );

  Iterable<Value> forward(Iterable<Value> inputs) {
    return [for (final neuron in neurons) neuron.forward(inputs)];
  }

  @override
  List<Value> get parameters => [
    for (final neuron in neurons) ...neuron.parameters,
  ];
}

class MultiLayerPerceptron extends Module {
  // Inputs size
  final int neuronInputsCount;

  // How many neurons per layer, the last one is how many outputs
  final Iterable<int> layerNeuronCounts;
  final Iterable<Layer> layers;

  MultiLayerPerceptron({
    required this.neuronInputsCount,
    required this.layerNeuronCounts,
  }) : layers = List.generate(
         layerNeuronCounts.length,
         (i) => Layer(
           neuronInputsCount:
               i == 0
                   ? neuronInputsCount
                   : layerNeuronCounts.elementAt(
                     i - 1,
                   ), // Previous layer's output count
           neuronsCount: layerNeuronCounts.elementAt(i),
         ),
       );

  Iterable<Value> forward(Iterable<Object> inputs) {
    var x = inputs.map((e) => e is num ? Value(e) : e).cast<Value>();

    for (final layer in layers) {
      x = layer.forward(x);
    }

    return x;
  }

  @override
  List<Value> get parameters => [
    for (final layer in layers) ...layer.parameters,
  ];
}
