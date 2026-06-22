"""
Orchestration primitives that are type-level no-ops: async DMA
(`copy_from_async` + semaphore passthrough) and `tpl.when`. The verifier
traces through them; the runtime emits the real `pltpu` ops.
"""
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim


def test_async_dma_with_semaphore(reset):
    """`copy_from_async(src, sem)` types the scratch immediately; runtime
    overlaps the DMA with the work between start and `.wait()`."""
    B = dim("AB", 8); D = dim("AD", 128); F = dim("AF", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), B, D, name="x")
    w = tjax.random.normal(jax.random.PRNGKey(1), D, F, name="w")

    def kernel(x_r, w_r, o_r, w_vmem, sem):
        dma = w_vmem.copy_from_async(w_r, sem)
        xv = x_r.load() * 2
        dma.wait()
        o_r.assign(tjax.einsum(xv, w_vmem.load(), "AB AD, AD AF -> AB AF"))

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("(2 * x:AB AD, w:AD AF -> AB AF)", (B, F), jnp.float32),
        in_specs=[pl.BlockSpec(), pl.BlockSpec(memory_space=pl.ANY)],
        scratch=[
            tpl.ScratchSpec("w_vmem", (128, 128), jnp.float32),
            pltpu.SemaphoreType.DMA,
        ],
    )(x, w)
    assert jnp.allclose(out.arr, (x.arr * 2) @ w.arr, atol=1e-4)


def test_axiom_after_raw_block(reset):
    """A raw-Pallas block writes a scratch ref directly; `.axiom(declared)`
    types it for downstream typed ops. Models the collective pattern: the
    raw block computes a device-sum the verifier can't see, and the typed
    code declares + composes around it."""
    DEV = dim("DEV", 4); B = dim("XB", 8); D = dim("XD", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), DEV, B, D, name="x")
    bias = tjax.random.normal(jax.random.PRNGKey(1), B, D, name="bias")

    def reference(x, bias):
        return x.sum(DEV) + bias

    def kernel(x_r, bias_r, o_r, ar_out):
        # --- raw-Pallas block (verifier sees nothing here) ---
        ar_out.ref[...] = x_r.ref[...].sum(axis=0)
        # --- declare its typed effect ---
        ar_out.axiom(x_r.load().sum(DEV))
        # --- typed composition continues ---
        o_r.assign(ar_out.load() + bias_r.load())

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(None, (B, D), jnp.float32),
        reference=reference,
        scratch=[tpl.ScratchSpec("ar_out", (8, 128), jnp.float32)],
    )(x, bias)
    assert jnp.allclose(out.arr, x.arr.sum(0) + bias.arr, atol=1e-4)


def test_axiom_declaration_is_what_verifier_sees(reset):
    """A wrong `.axiom()` declaration is caught downstream — the verifier
    checks against what was *declared*, not what the raw block wrote."""
    N = dim("XN", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32), N, name="x")

    def kernel(x_r, o_r, buf):
        buf.ref[...] = x_r.ref[...] * 2
        buf.axiom(x_r.load() * 3)  # wrong declaration
        o_r.assign(buf.load())

    import pytest as _pytest
    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * x:XN", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("buf", (128,), jnp.float32)],
    )
    with _pytest.raises(ValueError, match="does not match spec"):
        runner(x)


def test_typed_ref_at_index(reset):
    """`TypedRef.at(idx)` slices the ref and drops the leading dim from
    the ShapeType; the ET stays the original leaf (body-equivalence)."""
    L = dim("L", 4); B = dim("LB", 8); D = dim("LD", 128)
    w = tjax.random.normal(jax.random.PRNGKey(0), L, B, D, name="w")

    def kernel(w_r, o_r, buf):
        buf.copy_from(w_r.at(2))
        o_r.assign(buf.load() * 2)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * w:L LB LD", (B, D), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        scratch=[tpl.ScratchSpec("buf", (8, 128), jnp.float32)],
    )(w)
    assert jnp.allclose(out.arr, w.arr[2] * 2, atol=1e-5)


def test_fori_loop_ref_mutating(reset):
    """`tpl.fori_loop` body mutates a real scratch ref each iteration; the
    body's typed store is verified once, and the post-loop value is what
    `.axiom()` declares."""
    N = dim("FN", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32), N, name="x")

    def reference(x):
        return x * 4.0

    def kernel(x_r, o_r, acc):
        acc.store(x_r.load() * 0.0)

        def body(k):
            acc.store(acc.load() + x_r.load())

        tpl.fori_loop(0, 4, body)
        acc.axiom(x_r.load() * 4.0)
        o_r.assign(acc.load())

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(None, (N,), jnp.float32),
        reference=reference,
        scratch=[tpl.ScratchSpec("acc", (128,), jnp.float32)],
    )(x)
    assert jnp.allclose(out.arr, 4.0)


def test_when_gates_runtime_only(reset):
    """`tpl.when(raw_bool)` runtime-gates the body; verifier traces it."""
    N = dim("WN", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32) * 3, N, name="x")

    def kernel(x_r, o_r, tmp):
        tmp.store(x_r.load())

        @tpl.when(jnp.bool_(True))
        def _():
            tmp.store(x_r.load() * 2)

        o_r.assign(tmp.load())

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * x:WN", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("tmp", (128,), jnp.float32)],
    )(x)
    assert jnp.allclose(out.arr, 6.0)


def test_fori_loop_body_equivalence(reset):
    """`tpl.fori_loop(reference_body=, carries=)` rebinds carries to fresh
    leaves, traces body once, and checks each carry's output ET equals the
    reference's. Post-loop the carry is an opaque hash-leaf."""
    N = dim("FE_N", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32), N, name="x")

    def kernel(x_r, o_r, acc):
        acc.store(x_r.load() * 0.0)

        def ref_body(k, acc_v):
            return acc_v + x_r.load()

        def body(k):
            acc.store(x_r.load() + acc.load())

        tpl.fori_loop(0, 4, body, name="k", reference_body=ref_body, carries=(acc,))
        acc.axiom(x_r.load() * 4.0)
        o_r.assign(acc.load())

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("4 * x:FE_N", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("acc", (128,), jnp.float32)],
    )(x)
    assert jnp.allclose(out.arr, 4.0)


def test_fori_loop_body_equivalence_rejects_divergence(reset):
    """A body that doesn't match `reference_body` raises at the carry."""
    N = dim("FD_N", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32), N, name="x")

    def kernel(x_r, o_r, acc):
        acc.store(x_r.load() * 0.0)

        def ref_body(k, acc_v):
            return acc_v + x_r.load()

        def body(k):
            acc.store(acc.load() + x_r.load() * 2)

        tpl.fori_loop(0, 4, body, name="k", reference_body=ref_body, carries=(acc,))
        o_r.assign(acc.load())

    import pytest as _pytest
    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("x:FD_N", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("acc", (128,), jnp.float32)],
    )
    with _pytest.raises(Exception, match="does not match reference_body"):
        runner(x)


def test_when_domain_tags_carries(reset):
    """`tpl.when(layer > 0, tags=(carry,))` wraps the carry's type in
    `TagCond` after the body, and runtime-gates via the domain's
    `runtime_value` evaluation."""
    from stile.indexing import SymbolicInt
    N = dim("WT_N", 128)
    x = tjax.tensor(jnp.ones(128, dtype=jnp.float32), N, name="x")

    def kernel(x_r, o_r, c):
        c.store(x_r.load() * 0.0)

        def body(layer):
            @tpl.when(layer > 0, tags=(c,))
            def _():
                c.store(c.load() + x_r.load())

        tpl.fori_loop(0, 3, body, name="layer")
        c.axiom(x_r.load() * 2.0)
        o_r.assign(c.load())

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * x:WT_N", (N,), jnp.float32),
        scratch=[tpl.ScratchSpec("c", (128,), jnp.float32)],
    )(x)
    # layer 0: gated off (c stays 0); layers 1, 2: each adds x. Result = 2*x.
    assert jnp.allclose(out.arr, 2.0)
