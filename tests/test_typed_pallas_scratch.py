"""
TypedPallas VMEM-scratch refs — `TypedScratchRef` round-trips a
`TypedJaxArray`'s expression through an unverified store/load, so a
kernel that stages an intermediate through scratch (the way Pallas
kernels stage through VMEM) is still proven against its spec.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim


def test_scratch_roundtrip_verified(reset):
    """
    `tmp = x * 2; scratch.store(tmp); o = scratch.load() + 1` proves
    `2*x + 1` — the scratch hop is a no-op rename to the verifier.
    """
    N = dim("SCR_N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref, tmp):
        tmp.store(x_ref.load() * 2)
        o_ref.assign(tmp.load() + 1)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * SCR_N + 1", (N,), jnp.float32),
        scratch=("tmp",),
    )(x)

    assert jnp.allclose(result.arr, x.arr * 2 + 1)


def test_scratch_divergence_rejected(reset):
    """A wrong factor staged through scratch is still caught."""
    N = dim("SCR_W", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref, tmp):
        tmp.store(x_ref.load() * 3)
        o_ref.assign(tmp.load() + 1)

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * SCR_W + 1", (N,), jnp.float32),
        scratch=1,
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(x)


def test_rsqrt_normalizes_against_recip_sqrt(reset):
    """`tjax.rsqrt(x)` proves `1 / sqrt(x)` and runs `lax.rsqrt`."""
    N = dim("RSQ", 4)
    x = tjax.tensor(jnp.array([1.0, 4.0, 9.0, 16.0]), N, name="x")

    def kernel(x_ref, o_ref):
        o_ref.assign(tjax.rsqrt(x_ref.load()))

    out = tpl.typed_pallas_call(
        kernel, out_type=tpl.OutputSpec("1 / sqrt(x:RSQ)", (N,), jnp.float32)
    )(x)
    assert jnp.allclose(out.arr, 1.0 / jnp.sqrt(x.arr), atol=1e-6)


def test_scratch_load_before_store_raises(reset):
    N = dim("SCR_E", 4)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref, tmp):
        o_ref.assign(tmp.load())

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("SCR_E", (N,), jnp.float32),
        scratch=("tmp",),
    )
    with pytest.raises(ValueError, match="loaded before any store"):
        runner(x)
