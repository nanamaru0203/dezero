from .variable import Variable
from .utils import as_array, Config
import numpy as np
import weakref


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backdrop:
            self.inputs = inputs
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs = [weakref.ref(output) for output in outputs]
        if len(outputs) < 2:
            return outputs[0]
        return outputs

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        return gy * 2 * self.inputs[0].data


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return gy * np.exp(self.inputs[0].data)


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)
