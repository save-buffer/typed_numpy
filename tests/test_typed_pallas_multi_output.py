"""
Multi-output / untyped-passthrough / axiom typed-Pallas tests.

These cover the megakernel-facing extensions:

  - ``out_type`` may be a tuple of ``OutputSpec``: the kernel receives one
    ``TypedOutputRef`` per spec and the runner returns a tuple.
  - A non-``TypedJaxArray`` input is forwarded as a raw Pallas ref —
    the verifier never sees it.
  - ``tjax.axiom`` re-enters the typed world after an untyped block.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim


def test_two_outputs(reset):
    N = dim("N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def kernel(x_ref, a_ref, b_ref):
        v = x_ref.load()
        a_ref.assign(v * 2)
        b_ref.assign(v * 3)

    a, b = tpl.typed_pallas_call(
        kernel,
        out_type=(
            tpl.OutputSpec("2 * X:N", (N,), jnp.float32),
            tpl.OutputSpec("3 * X:N", (N,), jnp.float32),
        ),
    )(x)
    assert jnp.allclose(a.arr, x.arr * 2)
    assert jnp.allclose(b.arr, x.arr * 3)


def test_two_outputs_one_wrong_rejected(reset):
    N = dim("N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def kernel(x_ref, a_ref, b_ref):
        v = x_ref.load()
        a_ref.assign(v * 2)
        b_ref.assign(v * 4)  # spec says 3

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=(
            tpl.OutputSpec("2 * X:N", (N,), jnp.float32),
            tpl.OutputSpec("3 * X:N", (N,), jnp.float32),
        ),
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(x)


def test_untyped_input_passthrough(reset):
    """A raw ``jax.Array`` input arrives in the kernel as the bare Pallas
    ref; the typed math around it still verifies."""
    N = dim("N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")
    table = jnp.arange(N.size, dtype=jnp.int32)  # untyped

    def kernel(x_ref, table_ref, o_ref):
        assert isinstance(x_ref, tpl.TypedRef)
        assert not isinstance(table_ref, tpl.TypedRef)
        _ = table_ref[...]  # readable as a raw ref
        o_ref.assign(x_ref.load() * 2)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * X:N", (N,), jnp.float32),
    )(x, table)
    assert jnp.allclose(out.arr, x.arr * 2)


def test_axiom_reentry(reset):
    """An untyped block's result, wrapped via ``tjax.axiom``, composes
    with typed inputs and verifies against a spec that names it as a
    leaf."""
    N = dim("N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")
    raw = jax.random.normal(jax.random.PRNGKey(1), (N.size,))  # untyped

    def kernel(x_ref, raw_ref, o_ref):
        # "Collective" stand-in: an untyped read plus some untyped math.
        ar = tjax.axiom(raw_ref[...] + 0.0, N, name="AR")
        o_ref.assign(x_ref.load() + ar)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("X:N + AR:N", (N,), jnp.float32),
    )(x, raw)
    assert jnp.allclose(out.arr, x.arr + raw)


def test_fori_loop_body_equiv(reset):
    """``fori_loop(reference_body=...)`` proves the kernel body equals the
    reference body once, with symbolic ``k`` and a fresh-leaf carry; the
    kernel and the reference then produce the same output leaf so the outer
    typed_pallas_call verification passes."""
    N = dim("N", 8)
    n_iters = 4
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def ref_body(k, c):
        return c * 2.0 + 1.0

    def kernel(x_ref, o_ref):
        # Same recurrence, written differently (distributed).
        def body(k, c):
            return c * 2.0 + (c - c) + 1.0
        out = tjax.fori_loop(
            0, n_iters, body, x_ref.load(), reference_body=ref_body,
        )
        o_ref.assign(out)

    def reference(x):
        return tjax.fori_loop(
            0, n_iters, ref_body, x, reference_body=ref_body,
        )

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(None, (N,), jnp.float32),
        reference=reference,
    )(x)
    # 4 iterations of c -> 2c + 1 from x.
    expected = x.arr
    for _ in range(n_iters):
        expected = expected * 2 + 1
    assert jnp.allclose(out.arr, expected)


def test_fori_loop_body_equiv_rejects_mismatch(reset):
    N = dim("N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def ref_body(k, c):
        return c * 2.0

    with pytest.raises(AssertionError, match="does not match reference_body"):
        tjax.fori_loop(
            0, 4, lambda k, c: c * 3.0, x, reference_body=ref_body,
        )


def test_reference_multi_output(reset):
    """A single tjax reference supplies the expected expression for every
    output — no spec strings."""
    N = dim("N", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, name="b")

    def reference(a, b):
        return a + b, a * b

    def kernel(a_ref, b_ref, sum_ref, prod_ref):
        a, b = a_ref.load(), b_ref.load()
        sum_ref.assign(a + b)
        prod_ref.assign(a * b)

    s, p = tpl.typed_pallas_call(
        kernel,
        out_type=(
            tpl.OutputSpec(None, (N,), jnp.float32),
            tpl.OutputSpec(None, (N,), jnp.float32),
        ),
        reference=reference,
    )(a, b)
    assert jnp.allclose(s.arr, a.arr + b.arr)
    assert jnp.allclose(p.arr, a.arr * b.arr)


def test_reference_rejects_mismatch(reset):
    N = dim("N", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N, name="a")

    def kernel(a_ref, o_ref):
        o_ref.assign(a_ref.load() * 3)

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(None, (N,), jnp.float32),
        reference=lambda a: a * 2,
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(a)


def test_axiom_spec_form(reset):
    """``axiom(..., spec=...)`` declares a non-leaf expression."""
    N = dim("N", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    # Kernel multiplies by 2 once, then declares that result *is* 2*X
    # (axiom), then multiplies by 3 — verifier sees 6*X.
    def kernel(x_ref, o_ref):
        v = x_ref.load() * 2
        v = tjax.axiom(v.arr, N, spec="2 * X:N")
        o_ref.assign(v * 3)

    out = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("6 * X:N", (N,), jnp.float32),
    )(x)
    assert jnp.allclose(out.arr, x.arr * 6)
