"""
Real Pallas VMEM scratch via `ScratchSpec` + HBM input placement via
`in_specs`. The scratch ref is allocated by `pl.pallas_call(scratch_shapes=...)`;
`.store`/`.load` are real VMEM writes/reads, and `.copy_from(TypedRef)` is a
ref-to-ref copy that Mosaic lowers as a DMA. Under `interpret=True` the same
trace runs on CPU.
"""
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import pytest

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim


def test_real_scratch_roundtrip(reset):
    """`ScratchSpec` allocates a real ref; store/load go through it and the
    spec still verifies."""
    N = dim("RS_N", 128)
    x = tjax.tensor(jnp.arange(128, dtype=jnp.float32), N, name="x")

    def kernel(x_ref, o_ref, tmp):
        tmp.store(x_ref.load() * 2)
        o_ref.assign(tmp.load() + 1)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * x:RS_N + 1", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("tmp", (128,), jnp.float32)],
    )(x)
    assert jnp.allclose(out.arr, x.arr * 2 + 1)


def test_hbm_input_dma_into_scratch(reset):
    """Weight in HBM (`memory_space=pl.ANY`), `copy_from(TypedRef)` into a
    real VMEM scratch, matmul from scratch — verifies and runs."""
    B = dim("RB", 8); D = dim("RD", 128); HF = dim("RHF", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), B, D, name="x")
    w = tjax.random.normal(jax.random.PRNGKey(1), HF, D, name="w")

    def kernel(x_ref, w_ref, o_ref, w_vmem):
        w_vmem.copy_from(w_ref)
        o_ref.assign(tjax.einsum(x_ref.load(), w_vmem.load(), "RB RD, RHF RD -> RB RHF"))

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("(x:RB RD, w:RHF RD -> RB RHF)", (B, HF), jnp.float32),
        in_specs=[pl.BlockSpec(), pl.BlockSpec(memory_space=pl.ANY)],
        scratch=[tpl.ScratchSpec("w_vmem", (128, 128), jnp.float32)],
    )(x, w)
    assert jnp.allclose(out.arr, x.arr @ w.arr.T, atol=1e-4)


def test_mixed_real_and_holder_scratch(reset):
    """Real and bare-name scratch can be interleaved; declaration order is
    preserved in the kernel."""
    N = dim("RM_N", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32), N, name="x")

    def kernel(x_ref, o_ref, holder, real):
        holder.store(x_ref.load() * 3)
        real.store(holder.load() + 1)
        o_ref.assign(real.load())

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("3 * x:RM_N + 1", (N,), jnp.float32),
        scratch=["holder", tpl.ScratchSpec("real", (128,), jnp.float32)],
    )(x)
    assert jnp.allclose(out.arr, 4.0)


def test_untyped_scratch_spec_real_block(reset):
    """`UntypedScratchSpec` allocates one VMEM block; views are real
    slices, list-einsum reads the parent (no concatenate), and per-view
    DMAs land in the right slice."""
    B = dim("UB", 8); D = dim("UD", 128); HF = dim("UHF", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), B, D, name="x")
    w0 = tjax.random.normal(jax.random.PRNGKey(1), HF, D, name="w0")
    w1 = tjax.random.normal(jax.random.PRNGKey(2), HF, D, name="w1")

    def reference(x, w0, w1):
        return (
            tjax.einsum(x, w0, "UB UD, UHF UD -> UB UHF"),
            tjax.einsum(x, w1, "UB UD, UHF UD -> UB UHF"),
        )

    def kernel(x_r, w0_r, w1_r, o0, o1, buf):
        buf[0].copy_from(w0_r)
        buf[1].copy_from(w1_r)
        r0, r1 = tjax.einsum(x_r.load(), buf.loads(), "UB UD, UHF UD -> UB UHF")
        o0.assign(r0)
        o1.assign(r1)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=(
            tpl.OutputSpec(None, (B, HF), jnp.float32),
            tpl.OutputSpec(None, (B, HF), jnp.float32),
        ),
        reference=reference,
        in_specs=[pl.BlockSpec(), pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)],
        scratch=[tpl.UntypedScratchSpec("buf", parts=[("a", 128), ("b", 128)], trailing=(128,))],
    )(x, w0, w1)
    r0, r1 = out
    assert jnp.allclose(r0.arr, x.arr @ w0.arr.T, atol=1e-4)
    assert jnp.allclose(r1.arr, x.arr @ w1.arr.T, atol=1e-4)


def test_real_scratch_load_before_store_raises(reset):
    N = dim("RE_N", 8)
    x = tjax.tensor(jnp.ones(8, dtype=jnp.float32), N, name="x")

    def kernel(x_ref, o_ref, tmp):
        o_ref.assign(tmp.load())

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("RE_N", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("tmp", (8,), jnp.float32)],
    )
    with pytest.raises(ValueError, match="loaded before any store"):
        runner(x)
