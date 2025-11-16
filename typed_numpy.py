from dataclasses import dataclass

import numpy as np

a = np.random.randn(10, 10)
b = np.random.randn(10, 10)

@dataclass
class FullDim:
    name : str
    size : int

@dataclass
class Sliced:
    dim : "Dim"
    start : int | None
    end : int | None

Dim = FullDim | Sliced

def dim_contains(dim : Dim, target : FullDim):
    match dim:
        case FullDim(n, s):
            return n == target.name and s == target.size
        case Sliced(d, _, _):
            return dim_contains(d, target)

ExprType = Reduced | Add

@dataclass
class Reduced:
    dim : "Dim"
    start : int | None
    end : int | None

class Typed:
    def __init__(self, arr : np.ndarray, *dim_type, expr_type=None):
        self.arr = arr
        if len(dim_type) != len(self.arr.shape):
            raise ValueError("Number of attributes must match physical dimension")
        self.dim_type = dim_type
        self.expr_type = expr_type

    def slice(self, dim : FullDim, start : int | None, end : int | None):
        slice_expr = []
        dim_type = []
        expr_type = self.expr_type

        for d in self.dim_type:
            if dim_contains(d, dim):
                slice_expr.append(slice(start, end))
                dim_type.append(Sliced(d, start, end))
            else:
                slice_expr.append(slice(None))
                dim_type.append(d)

        return Typed(self.arr[*slice_expr], *dim_type, expr_type=expr_type)

    @property
    def type(self):
        return self.dim_type

    @property
    def shape(self):
        return self.arr.shape

def reduce(a : Typed, b : Typed, dim : FullDim) -> Typed:
    pass

M, N, K = FullDim('M', 10), FullDim('N', 10), FullDim('K', 10)

a = Typed(a, M, N)
b = Typed(b, N, K)

a_sliced = a.slice(M, 0, 5).slice(N, 0, 5)
b_sliced = b.slice(N, 0, 5).slice(K, 0, 5)

print("Shape:", a_sliced.shape)
print("Type: ", b_sliced.type)
