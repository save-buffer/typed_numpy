"""
`tpl.untyped_scratch` + list-form `tjax.einsum`: a flat VMEM buffer carved
into contiguous typed views, fed as a list to one einsum that lowers to a
single coalesced matmul. Type-level: one independent ET per view, so each
output proves equal to the per-view einsum.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim


def test_list_einsum_coalesced(reset):
    B = dim("B", 4); D = dim("D", 8); HF = dim("HF", 3)
    x = tjax.random.normal(jax.random.PRNGKey(0), B, D, name="x")
    w0 = tjax.random.normal(jax.random.PRNGKey(1), HF, D, name="w0")
    w1 = tjax.random.normal(jax.random.PRNGKey(2), HF, D, name="w1")

    def reference(x, w0, w1):
        return (
            tjax.einsum(x, w0, "B D, HF D -> B HF"),
            tjax.einsum(x, w1, "B D, HF D -> B HF"),
        )

    def kernel(x_r, w0_r, w1_r, o0, o1):
        vmem = tpl.untyped_scratch("w", parts=[("a", HF.size), ("b", HF.size)])
        vmem["a"].copy_from(w0_r.load())
        vmem["b"].copy_from(w1_r.load())
        r0, r1 = tjax.einsum(x_r.load(), vmem.loads(), "B D, HF D -> B HF")
        o0.assign(r0)
        o1.assign(r1)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=(
            tpl.OutputSpec(None, (B, HF), jnp.float32),
            tpl.OutputSpec(None, (B, HF), jnp.float32),
        ),
        reference=reference,
    )(x, w0, w1)
    r0, r1 = out
    assert jnp.allclose(r0.arr, x.arr @ w0.arr.T, atol=1e-5)
    assert jnp.allclose(r1.arr, x.arr @ w1.arr.T, atol=1e-5)


def test_list_einsum_noncontiguous_fallback(reset):
    """Without `_slot` metadata (plain TypedJaxArrays), list-einsum falls
    back to per-entry matmuls; type-level result is unchanged."""
    B = dim("B", 2); D = dim("D", 4); HF = dim("HF", 3)
    x = tjax.random.normal(jax.random.PRNGKey(0), B, D, name="x")
    ws = [tjax.random.normal(jax.random.PRNGKey(i + 1), HF, D, name=f"w{i}") for i in range(2)]
    outs = tjax.einsum(x, ws, "B D, HF D -> B HF")
    for o, w in zip(outs, ws):
        assert jnp.allclose(o.arr, x.arr @ w.arr.T, atol=1e-5)


def test_list_einsum_rejects_contracted_lead(reset):
    """Coalesced lowering requires y's leading dim to be free."""
    B = dim("B", 2); D = dim("D", 4)
    x = tjax.random.normal(jax.random.PRNGKey(0), B, D, name="x")
    w = tjax.random.normal(jax.random.PRNGKey(1), D, B, name="w")
    vmem = tpl.untyped_scratch("w", parts=[("a", D.size)])
    vmem["a"].copy_from(w)
    with pytest.raises(ValueError, match="leading dim to be free"):
        tjax.einsum(x, vmem.loads(), "B D, D B -> B")
