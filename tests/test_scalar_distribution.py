"""Scalar-times-sum distribution in the normalizer.

``Product(const=c, factors={Sum})`` flattens into the parent sum as
``c*child`` for each child — the size-preserving scalar case of the
distributive law. Tensor-times-sum was already handled by the
product-over-sum machinery; the scalar-only case fell through
``_as_single_factor``'s ``const == 1.0`` guard.
"""
import stile.jax as tjax
from stile import dim
from stile.verification import verify_exprs_equivalent


def test_scalar_distributes_over_sum(reset):
    N = dim("N", 4)
    a = tjax.tensor(None, N, name="a")
    b = tjax.tensor(None, N, name="b")
    c = tjax.tensor(None, N, name="c")
    assert verify_exprs_equivalent(
        (0.25 * (a + b) + c).type.et, (c + 0.25 * b + 0.25 * a).type.et
    )


def test_scalar_distribution_enables_cancellation(reset):
    N = dim("N", 4)
    a = tjax.tensor(None, N, name="a")
    b = tjax.tensor(None, N, name="b")
    assert verify_exprs_equivalent(
        (0.5 * (a + b) - 0.5 * a).type.et, (0.5 * b).type.et
    )
