import 'dart:math';

enum ValueOperation { add, subtract, multiply, divide, tanh, power, sigmoid }

class Value {
  final Set<Value> children;
  final ValueOperation? operation;

  Value(this.data, [this.children = const {}, this.operation]);

  void Function()? localBackward;
  String? label;
  num grad = 0;
  num data;

  void backward() {
    final topologicalGraph = <Value>[];
    final visited = <Value>{};

    buildTopologicalGraph(Value value) {
      if (visited.contains(value)) return;
      visited.add(value);
      for (final child in value.children) {
        buildTopologicalGraph(child);
      }
      topologicalGraph.add(value);
    }

    buildTopologicalGraph(this);

    grad = 1.0;
    for (final value in topologicalGraph.reversed) {
      value.localBackward?.call();
    }
  }

  @override
  String toString() {
    final parameters = [
      'data=${data.toStringAsFixed(4)}',
      if (operation != null) 'operation=${operation?.name}',
      if (label?.isNotEmpty == true) 'label=$label',
      'grad=${grad.toStringAsFixed(4)}',
      if (children.isNotEmpty) 'children=$children',
    ];

    return 'Value(${parameters.join(', ')})';
  }

  Map<String, dynamic> toJson() {
    return {
      'data': data,
      'operation': operation?.name,
      'label': label,
      'grad': grad,
      'children': children.map((child) => child.toJson()).toList(),
    };
  }

  Value _fromObject(Object object) {
    return switch (object.runtimeType) {
          const (num) => Value(object as num),
          const (int) => Value(object as int),
          const (double) => Value(object as double),
          const (Value) => object,
          _ => throw TypeError(),
        }
        as Value;
  }

  operator +(Object other) {
    final effectiveOther = _fromObject(other);
    final out = Value(data + effectiveOther.data, {
      this,
      effectiveOther,
    }, ValueOperation.add);

    out.localBackward = () {
      grad += out.grad;
      effectiveOther.grad += out.grad;
    };

    return out;
  }

  operator -(Object other) {
    final effectiveOther = _fromObject(other);
    final out = Value(data - effectiveOther.data, {
      this,
      effectiveOther,
    }, ValueOperation.subtract);

    out.localBackward = () {
      grad += out.grad;
      effectiveOther.grad -= out.grad;
    };

    return out;
  }

  operator *(Object other) {
    final effectiveOther = _fromObject(other);

    final out = Value(data * effectiveOther.data, {
      this,
      effectiveOther,
    }, ValueOperation.multiply);

    out.localBackward = () {
      grad += out.grad * effectiveOther.data;
      effectiveOther.grad += out.grad * data;
    };

    return out;
  }

  operator /(Object other) {
    final effectiveOther = _fromObject(other);

    final out = Value(data / effectiveOther.data, {
      this,
      effectiveOther,
    }, ValueOperation.divide);

    out.localBackward = () {
      grad += out.grad * pow(effectiveOther.data, -1);
      effectiveOther.grad += out.grad * (-data * pow(effectiveOther.data, -2));
    };

    return out;
  }

  Value tanh() {
    final t = (exp(2 * data) - 1) / (exp(2 * data) + 1);
    final out = Value(t, {this}, ValueOperation.tanh);
    out.localBackward = () => grad += (1 - pow(t, 2)) * out.grad;
    return out;
  }

  Value sigmoid() {
    final t = (1 / (1 + exp(-data)));
    final out = Value(t, {this}, ValueOperation.sigmoid);
    out.localBackward = () => grad += t * (1 - t) * out.grad;
    return out;
  }

  Value power(Object exponent) {
    final effectiveExponent = _fromObject(exponent);
    final out = Value(pow(data, effectiveExponent.data), {
      this,
    }, ValueOperation.power);
    out.localBackward =
        () =>
            grad +=
                out.grad *
                effectiveExponent.data *
                pow(data, effectiveExponent.data - 1);

    return out;
  }

  Value exponent() {
    final out = Value(exp(data), {this}, ValueOperation.power);
    out.localBackward = () => grad += out.grad * out.data;

    return out;
  }
}
