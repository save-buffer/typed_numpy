import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import einops

from .type_nodes import *
from .egraph import Egraph, EclassID, check_if_exprs_equal_rust
from .specification import parse_spec_into_type

class Typed:
    def __init__(self, arr : np.ndarray, *dim_type : Dim, expr_type : ExprType | None = None):
        self.arr = arr
        if len(dim_type) != len(self.arr.shape):
            raise ValueError("Number of attributes must match physical dimension")
        self.dim_type : DimType = dim_type
        self.expr_type : ExprType = (
            expr_type
            if expr_type is not None
            else Tensor(tuple(dim_full_dim(d) for d in self.dim_type))
        )

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

    def repeat(self, dim : Dim) -> "Typed":
        new_dim_type = [dim]
        for d in self.dim_type:
            if dim_name(dim) == dim_name(d):
                raise ValueError(f"Cannot repeat a dim that already exists ({dim=})")
            new_dim_type.append(d)

        nrepeats = dim_size(dim)
        repeated_arr = self.arr[None].repeat(nrepeats, axis=0)
        new_expr_type = Repeat(dim_full_dim(dim), self.expr_type)
        return Typed(repeated_arr, *new_dim_type, expr_type=new_expr_type)

    def rearrange(self, *dims : Dim) -> "Typed":
        dims = tuple(dim_full_dim(d) for d in dims)

        dims_by_name = {}
        lhs_str = ""
        for d in self.dim_type:
            name = dim_name(d)
            dims_by_name[name] = d
            lhs_str += f"{name} "

        new_dim_type = []
        rhs_str = ""
        names = [dim_name(d) for d in dims]
        for n in names:
            if n not in dims_by_name:
                raise ValueError(f"Trying to rearrange with unknown dim {n}")
            new_dim_type.append(dims_by_name[n])
            rhs_str += f"{n} "

        new_arr = einops.rearrange(self.arr, f"{lhs_str} -> {rhs_str}")
        return Typed(new_arr, *new_dim_type, expr_type=self.expr_type)

    def reduce(self, op : ReduceOpType, dim : Dim) -> "Typed":
        dim = dim_full_dim(dim)

        new_dim_type = []
        reduction_dim = None
        ireduction_dim = None
        for i, d in enumerate(self.dim_type):
            if dim_name(dim) != dim_name(d):
                new_dim_type.append(d)
            else:
                ireduction_dim = i
                reduction_dim = d
        if reduction_dim is None:
            raise ValueError(f"Unknown reduction dimension {d}")
        assert ireduction_dim is not None
        
        new_expr_type = Reduce(
            op=op,
            dim=reduction_dim,
            child=self.expr_type
        )
        match op:
            case "sum":
                new_arr = self.arr.sum(axis=ireduction_dim)
            case "max":
                new_arr = self.arr.max(axis=ireduction_dim)

        return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

    def sum(self, dim : Dim) -> "Typed":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "Typed":
        return self.reduce("max", dim)

    def binary_op(self, other, op : BinaryOpType) -> "Typed":
        return _binary_op_helper(self, other, op)

    def __add__(self, other) -> "Typed":
        return self.binary_op(other, "+")

    def __sub__(self, other) -> "Typed":
        return self.binary_op(other, "-")

    def __mul__(self, other) -> "Typed":
        return self.binary_op(other, "*")

    def __truediv__(self, other) -> "Typed":
        return self.binary_op(other, "/")

    def __radd__(self, other) -> "Typed":
        return _binary_op_helper(other, self, "+")

    def __rsub__(self, other) -> "Typed":
        return _binary_op_helper(other, self, "-")

    def __rmul__(self, other) -> "Typed":
        return _binary_op_helper(other, self, "*")

    def __rtruediv__(self, other) -> "Typed":
        return _binary_op_helper(other, self, "/")

    def __matmul__(self, other) -> "Typed":
        return einsum(self, other, "M N, N K -> M K")

    def set(self, other : "Typed"):
        if self.dim_type != other.dim_type:
            raise ValueError("Only assignment between tiles of the same dim type is allowed")
        self.arr = other.arr
        self.expr_type = other.expr_type

    @property
    def type(self):
        return self.dim_type

    @property
    def shape(self):
        return self.arr.shape

def _binary_op_helper(slf, other, op):
    match slf, other:
        case Typed(), Typed():
            if slf.dim_type != other.dim_type:
                raise ValueError("Binary operations can only occur between tensors with the same shapes")
            match op:
                case "+":
                    new_arr = slf.arr + other.arr
                case "-":
                    new_arr = slf.arr - other.arr
                case "*":
                    new_arr = slf.arr * other.arr
                case "/":
                    new_arr = slf.arr / other.arr
                case "max":
                    new_arr = np.maximum(slf.arr, other.arr)

            new_dim_type = slf.dim_type
            new_expr_type = BinaryOp(
                op=op,
                lhs=slf.expr_type,
                rhs=other.expr_type,
            )
            return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)
        case Typed(), x:
            match op:
                case "+":
                    new_arr = slf.arr + x
                case "-":
                    new_arr = slf.arr - x
                case "*":
                    new_arr = slf.arr * x
                case "/":
                    new_arr = slf.arr / x
                case "max":
                    new_arr = np.maximum(slf.arr, x)

            new_dim_type = slf.dim_type
            new_expr_type = BinaryOp(
                op=op,
                lhs=slf.expr_type,
                rhs=Constant(x),
            )
            return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)
        case x, Typed():
            match op:
                case "+":
                    new_arr = x + other.arr
                case "-":
                    new_arr = x - other.arr
                case "*":
                    new_arr = x * other.arr
                case "/":
                    new_arr = x / other.arr
                case "max":
                    new_arr = np.maximum(x, other.arr)

            new_dim_type = other.dim_type
            new_expr_type = BinaryOp(
                op=op,
                lhs=Constant(x),
                rhs=other.expr_type,
            )
            return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)
            

def einsum(a : Typed, b : Typed, einstr : str) -> Typed:
    lhs, rhs = einstr.split('->')
    a_str, b_str = lhs.split(',')
    a_dims = a_str.strip().split(' ')
    b_dims = b_str.strip().split(' ')
    rhs_dim_names = rhs.strip().split(' ')

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
    a_repeated = a
    for d in b.dim_type:
        name = dim_name(d)
        if name not in common_dims:
            a_repeated = a_repeated.repeat(d)

    b_repeated = b
    for d in a.dim_type:
        name = dim_name(d)
        if name not in common_dims:
            b_repeated = b_repeated.repeat(d)

    a, b = a_repeated, b_repeated

    reduction_dims = []
    for d in common_dims:
        if d not in rhs_dim_names:
            if a_dims_by_name[d] != b_dims_by_name[d]:
                raise ValueError(f"Trying to reduce along mismatching slices of dimension {d}: A={a_dims_by_name[d]}, B={b_dims_by_name[d]}")
            reduction_dims.append(a_dims_by_name[d])

    rhs_dims = []
    for d in rhs_dim_names:
        if d in a_dims_by_name:
            rhs_dims.append(a_dims_by_name[d])
        elif d in b_dims_by_name:
            rhs_dims.append(b_dims_by_name[d])

    a = a.rearrange(*rhs_dims, *reduction_dims)
    b = b.rearrange(*rhs_dims, *reduction_dims)

    c = a * b

    for d in reduction_dims:
        c = c.sum(d)

    return c

def exp(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="exp",
        child=x.expr_type,
    )
    new_arr = np.exp(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

def sin(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="sin",
        child=x.expr_type,
    )
    new_arr = np.sin(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

def cos(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="cos",
        child=x.expr_type,
    )
    new_arr = np.cos(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)
    
def sqrt(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="sqrt",
        child=x.expr_type,
    )
    new_arr = np.sqrt(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

def max(x : Typed | float, y : Typed | float) -> Typed:
    return _binary_op_helper(x, y, "max")

def expr_types_are_equivalent(
    dim_type : tuple[Dim, ...],
    expected : ExprType,
    actual : ExprType,
    niters : int = 10,
) -> bool:
    return check_if_exprs_equal_rust(expected, actual)

def expr_simplifies(
    expr : Typed,
    spec : str,
    niters : int = 15,
) -> bool:
    spec_dt, spec_et = parse_spec_into_type(spec)
    return check_if_exprs_equal_rust(expr.expr_type, spec_et)

def rewrite_found(
    expr : Typed,
    rewrite : str,
    niters : int = 5,
):
    rw_dt, rw_et = parse_spec_into_type(rewrite)
    return check_if_exprs_equal_rust(expr.expr_type, rw_et)

class TypedResult:
    def __init__(self, spec : str):
        self.expected_dim_type, self.expected_expr_type = parse_spec_into_type(spec)
        self.shape = tuple(dim_size(d) for d in self.expected_dim_type) if self.expected_dim_type is not None else tuple()
        self.arr = np.zeros(self.shape)
        
    def assign(self, result : Typed):
        if not expr_types_are_equivalent(
                dim_type=result.dim_type,
                expected=self.expected_expr_type,
                actual=result.expr_type,
        ):
            raise ValueError(f"Attempted to assign a tensor that does not match the spec! Expected: {self.expected_expr_type}, actual: {result.expr_type}")

        slice_expr = []
        for d in result.dim_type:
            ds, de = dim_start(d), dim_end(d)
            slice_expr.append(slice(ds, de))
        self.arr[*slice_expr] = result.arr

def reset_stile():
    g_dim_registry.clear()
