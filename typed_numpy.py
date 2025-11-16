from dataclasses import dataclass

import numpy as np
import einops

a = np.random.randn(10, 10)
b = np.random.randn(10, 10)

# TODO: Prevent creating multiple dims with the same name
@dataclass
class FullDim:
    name : str
    size : int

@dataclass
class Sliced:
    dim : FullDim
    start : int
    end : int

Dim = FullDim | Sliced

def dim_name(dim : Dim):
    match dim:
        case FullDim(n, _):
            return n
        case Sliced(d, _, _):
            return dim_name(d)

def dim_contains(dim : Dim, target : FullDim):
    match dim:
        case FullDim(n, s):
            return n == target.name and s == target.size
        case Sliced(d, _, _):
            return dim_contains(d, target)

def simplify_dim(dim : Dim) -> Dim:
    match dim:
        case FullDim(n, sz):
            return dim
        case Sliced(d, st, en):
            child = simplify_dim(d)
            match child:
                case FullDim(n, sz):
                    return Sliced(child, st, en)
                case Sliced(full, child_st, child_en):
                    # TODO: Check bounds
                    length = en - st
                    return Sliced(full, child_st + st, child_st + st + length)

BinaryOpType = Literal["+", "-", "*", "/"]

@dataclass
class BinaryOp:
    lhs : "ExprType"
    rhs : "ExprType"
    dim : Dim
    op : BinaryOpType

@dataclass
class Reduced:
    child : "ExprType"
    dim : Dim
    start : int
    end : int

ExprType = Reduced | BinaryOp | None

class Typed:
    def __init__(self, arr : np.ndarray, *dim_type : Dim, expr_type=None):
        self.arr = arr
        if len(dim_type) != len(self.arr.shape):
            raise ValueError("Number of attributes must match physical dimension")
        self.dim_type = dim_type
        self.expr_type = expr_type

    def slice(self, dim : FullDim, start : int | None, end : int | None) -> "Typed":
        slice_expr = []
        dim_type = []
        expr_type = self.expr_type

        st : int = start if start is not None else 0

        dim_found = False
        for i, d in enumerate(self.dim_type):
            if dim_contains(d, dim):
                slice_expr.append(slice(start, end))

                dim_type.append(
                    Sliced(
                        d,
                        st,
                        end if end is not None else self.arr.shape[i] - st,
                    )
                )
                dim_found = True
            else:
                slice_expr.append(slice(None))
                dim_type.append(d)

        if not dim_found:
            raise ValueError(f"Invalid dim {dim}")

        for i in range(len(dim_type)):
            dim_type[i] = simplify_dim(dim_type[i])

        return Typed(self.arr[*slice_expr], *dim_type, expr_type=expr_type)

    @property
    def type(self):
        return self.dim_type

    @property
    def shape(self):
        return self.arr.shape

def reduce(a : Typed, b : Typed, einstr : str) -> Typed:
    lhs, rhs = einstr.split('->')
    a_str, b_str = lhs.split(',')
    a_dims = a_str.strip().split(' ')
    b_dims = b_str.strip().split(' ')
    rhs_dims = rhs.strip().split(' ')

    a_dims_by_name = {}
    b_dims_by_name = {}
    for d in a.dim_type:
        name = dim_name(d)
        if name not in a_dims:
            raise ValueError(f"Dimension {name} not found in tensor A's einsum string")
        a_dims_by_name[name] = d

    for d in b.dim_type:
        name = dim_name(d)
        if name not in b_dims:
            raise ValueError(f"Dimension {name} not found in tensor B's einsum string")
        b_dims_by_name[name] = d

    common_dims = a_dims_by_name.keys() & b_dims_by_name.keys()
    expr_type = a.expr_type
    for d in common_dims:
        if d not in rhs_dims:
            if a_dims_by_name[d] != b_dims_by_name[d]:
                raise ValueError(f"Trying to reduce along mismatching slices of dimension {d}: A={a_dims_by_name[d]}, B={b_dims_by_name[d]}")
            
            reduction_dims.add(d)
            
    result_dim_type = []
    for d in rhs_dims:
        assert d in a_dims_by_name or d in b_dims_by_name
        if d in a_dims_by_name and d in b_dims_by_name:
            if a_dims_by_name[d] != b_dims_by_name[d]:
                raise ValueError("Batch dimension of einsum is operating on mismatching slices! A={a_dims_by_name[d]}, B={b_dims_by_name[d]}")
            result_dim_type.append(a_dims_by_name[d])
        elif d in a_dims_by_name:
            result_dim_type.append(a_dims_by_name[d])
        elif d in b_dims_by_name:
            result_dim_type.append(b_dims_by_name[d])

    result = einops.einsum(a.arr, b.arr, einstr)

M, N, K = FullDim('M', 10), FullDim('N', 10), FullDim('K', 10)

a = Typed(a, M, N)
b = Typed(b, N, K)

a_sliced = a.slice(M, 0, 5).slice(N, 0, 5)
b_sliced = b.slice(N, 0, 5).slice(K, 0, 5)

print("Shape:", a_sliced.shape)
print("Type: ", a_sliced.type)

c_tile0 = reduce(a_sliced, b_sliced, "M N, N K -> M K")

