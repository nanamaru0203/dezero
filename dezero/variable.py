import numpy as np
import heapq


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, creator):
        self.creator = creator
        self.generation = creator.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)
                heapq.heappush(funcs, (-f.generation, id(f), f))

        add_func(self.creator)

        while funcs:
            _, _, f = heapq.heappop(funcs)
            x, y = f.inputs, f.outputs
            gys = [output().grad for output in y]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def clearglad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"
