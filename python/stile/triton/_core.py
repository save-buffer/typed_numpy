try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError(
        "Triton support requires the triton extra: pip install stile[triton]"
    ) from None

import stile.type as t
from ..type import *
from ..specification import parse_spec_into_type
from ..verification import verify_types_equivalent, verify_exprs_equivalent


class TypedTile:
    """A tile of data loaded from a TypedPointer, with tracked type info."""

    def __init__(self, data, type : Type):
        self.data = data
        self.type = type

    def __add__(self, other) -> "TypedTile":
        return _binary_op_helper(self, other, "+")

    def __sub__(self, other) -> "TypedTile":
        return _binary_op_helper(self, other, "-")

    def __mul__(self, other) -> "TypedTile":
        return _binary_op_helper(self, other, "*")

    def __truediv__(self, other) -> "TypedTile":
        return _binary_op_helper(self, other, "/")

    def __radd__(self, other) -> "TypedTile":
        return _binary_op_helper(other, self, "+")

    def __rsub__(self, other) -> "TypedTile":
        return _binary_op_helper(other, self, "-")

    def __rmul__(self, other) -> "TypedTile":
        return _binary_op_helper(other, self, "*")

    def __rtruediv__(self, other) -> "TypedTile":
        return _binary_op_helper(other, self, "/")

    def reduce(self, op : ReduceOpType, dim : Dim) -> "TypedTile":
        new_type = self.type.reduce(op, dim)

        for i, d in enumerate(self.type.dt):
            if dim_name(dim) == dim_name(d):
                ireduction_dim = i
                break

        match op:
            case "sum":
                new_data = tl.sum(self.data, axis=ireduction_dim)
            case "max":
                new_data = tl.max(self.data, axis=ireduction_dim)

        return TypedTile(new_data, new_type)

    def sum(self, dim : Dim) -> "TypedTile":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "TypedTile":
        return self.reduce("max", dim)

    def assert_equivalent(self, spec : str, *dim_override : Dim):
        expected_type = parse_spec_into_type(spec)
        expected_type = override_dims_in_type(expected_type, *dim_override)
        are_equivalent = verify_exprs_equivalent(
            expected_type.et,
            self.type.et,
        )
        assert are_equivalent


def _binary_op_helper(
    slf : TypedTile | float,
    other : TypedTile | float,
    op : BinaryOpType,
) -> TypedTile:
    lhs_type = slf.type if isinstance(slf, TypedTile) else slf
    rhs_type = other.type if isinstance(other, TypedTile) else other
    new_type = type_from_binary_op(lhs_type, rhs_type, op)

    lhs = slf.data if isinstance(slf, TypedTile) else slf
    rhs = other.data if isinstance(other, TypedTile) else other

    match op:
        case "+":
            new_data = lhs + rhs
        case "-":
            new_data = lhs - rhs
        case "*":
            new_data = lhs * rhs
        case "/":
            new_data = lhs / rhs
        case "max":
            new_data = tl.maximum(lhs, rhs)
        case _:
            raise ValueError(f"Unknown op {op}")

    return TypedTile(new_data, new_type)


def exp(x : TypedTile) -> TypedTile:
    new_type = t.exp(x.type)
    new_data = tl.exp(x.data)
    return TypedTile(new_data, new_type)


def sin(x : TypedTile) -> TypedTile:
    new_type = t.sin(x.type)
    new_data = tl.sin(x.data)
    return TypedTile(new_data, new_type)


def cos(x : TypedTile) -> TypedTile:
    new_type = t.cos(x.type)
    new_data = tl.cos(x.data)
    return TypedTile(new_data, new_type)


def sqrt(x : TypedTile) -> TypedTile:
    new_type = t.sqrt(x.type)
    new_data = tl.sqrt(x.data)
    return TypedTile(new_data, new_type)


def maximum(x : TypedTile, y : TypedTile) -> TypedTile:
    return _binary_op_helper(x, y, "max")


class TypedPointer:
    """
    A pointer to tensor data along with its type information.

    This is passed to Triton kernels and used to load/store typed tiles.
    """

    def __init__(self, ptr, type : Type, strides : tuple[int, ...]):
        self.ptr = ptr
        self.type = type
        self.strides = strides

    def tile_slice(self, dim : FullDim, start : int, end : int) -> "TypedPointer":
        """
        Return a TypedPointer to a slice of this tensor along the given dimension.
        Used to compute the pointer for a specific tile.
        """
        new_type = self.type.slice(dim, start, end)

        # Find the axis for this dimension and compute pointer offset
        offset = 0
        for i, d in enumerate(self.type.dt):
            if dim_contains(d, dim):
                offset = start * self.strides[i]
                break

        return TypedPointer(self.ptr + offset, new_type, self.strides)


def load(ptr : TypedPointer, offsets, mask=None) -> TypedTile:
    """
    Load a tile from a TypedPointer.

    Args:
        ptr: A TypedPointer with type information
        offsets: The offsets to load (from tl.arange, etc.)
        mask: Optional mask for out-of-bounds handling

    Returns:
        TypedTile with the loaded data and corresponding type
    """
    if mask is not None:
        data = tl.load(ptr.ptr + offsets, mask=mask)
    else:
        data = tl.load(ptr.ptr + offsets)

    return TypedTile(data, ptr.type)


def store(ptr : TypedPointer, tile : TypedTile, offsets, mask=None):
    """
    Store a TypedTile to a TypedPointer.

    Args:
        ptr: A TypedPointer with expected type information
        tile: The TypedTile to store
        offsets: The offsets to store at
        mask: Optional mask for out-of-bounds handling
    """
    if not verify_types_equivalent(ptr.type, tile.type):
        raise ValueError(
            f"Type mismatch in store! "
            f"Expected: {ptr.type}, got: {tile.type}"
        )

    if mask is not None:
        tl.store(ptr.ptr + offsets, tile.data, mask=mask)
    else:
        tl.store(ptr.ptr + offsets, tile.data)


def make_typed_pointer(tensor, type : Type) -> TypedPointer:
    """
    Create a TypedPointer from a tensor (torch.Tensor or similar) and its type.

    This should be called outside the kernel to wrap tensors before passing them in.
    """
    # Get strides from the tensor
    if hasattr(tensor, 'stride'):
        strides = tensor.stride()
    else:
        # Assume contiguous layout
        strides = None

    # Get the data pointer
    ptr = tensor

    return TypedPointer(ptr, type, strides)
