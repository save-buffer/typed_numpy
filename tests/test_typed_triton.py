import os
import sys
import platform

# Enable interpreter mode for testing without GPU
os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

import stile.triton as st

import pytest
from stile import dim, reset_stile
from stile.type import Type, Tensor, FullDim

@pytest.fixture
def reset():
    yield
    reset_stile()


@triton.jit
def add_kernel_typed(
    x_ptr, x_type_dt, x_type_et,
    y_ptr, y_type_dt, y_type_et,
    output_ptr, output_type_dt, output_type_et,
    n_elements,
    BLOCK_SIZE : tl.constexpr,
):
    """A typed vector addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load with type tracking
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def test_simple_add(reset):
    """Test basic typed vector addition."""
    N = dim('N', 1024)

    # Create tensors
    x = torch.randn(N.size)
    y = torch.randn(N.size)
    output = torch.empty_like(x)

    # Create types
    x_type = Type(dt=(N,), et=Tensor((N,)))
    y_type = Type(dt=(N,), et=Tensor((N,)))

    # Launch kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel_typed[grid](
        x, None, None,
        y, None, None,
        output, None, None,
        n_elements,
        BLOCK_SIZE=256,
    )

    # Verify numerical correctness
    expected = x + y
    assert torch.allclose(output, expected)


@triton.jit
def exp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr,
):
    """Exponential kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_exp(reset):
    """Test exponential operation."""
    N = dim('N', 512)

    x = torch.randn(N.size)
    output = torch.empty_like(x)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    exp_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=128,
    )

    expected = torch.exp(x)
    assert torch.allclose(output, expected)


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr,
):
    """
    Simple softmax kernel (processes entire vector in one block).
    This is a simplified version for testing.
    """
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the entire vector
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # Compute softmax with numerical stability
    x_max = tl.max(x, axis=0)
    x_stable = x - x_max
    exp_x = tl.exp(x_stable)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    tl.store(output_ptr + offsets, softmax, mask=mask)


def test_softmax(reset):
    """Test softmax kernel."""
    N = dim('N', 64)

    x = torch.randn(N.size)
    output = torch.empty_like(x)

    # For simplicity, process entire vector in one block
    softmax_kernel[(1,)](
        x,
        output,
        N.size,
        BLOCK_SIZE=64,
    )

    # Compare with PyTorch softmax
    expected = torch.softmax(x, dim=0)
    assert torch.allclose(output, expected, atol=1e-5)


tests = [
    test_simple_add,
    test_exp,
    test_softmax,
]

if __name__ == '__main__':
    if platform.system() != "Linux":
        print("Skipping Triton tests: only supported on Linux")
        sys.exit(0)

    for test in tests:
        print("Running", test)
        sys.stdout.flush()
        reset_stile()
        test(None)
    print("All Triton tests passed!")
