"""
TypedPallas — `stile.jax.pallas` is the typed wrapper around
`jax.experimental.pallas`. Same type discipline as `stile.jax`: every
value carries a `Type` (ShapeType + ExprType), and a per-assignment
verifier proves the kernel matches a one-line spec.

Distinct from `stile.jax`'s `TypedResult` because Pallas kernels run
inside a JIT trace where inputs and outputs are `Ref`s (mutable cells)
rather than values. The discipline:

  - Inputs and outputs are wrapped as `TypedRef` / `TypedOutputRef`.
  - `.load()` reads a `Ref` and produces a `TypedJaxArray` carrying the
    same `Type`. Once you've loaded, all the `tjax` ops compose normally
    — slice, einsum, exp, sum, where, mask, fori_loop.
  - `.assign(value)` on the output ref runs the verifier against the
    `OutputSpec` and stores. Same `verify_types_equivalent` machinery
    `tjax.TypedResult.assign` uses.

For local development, `interpret=True` runs Pallas on CPU with the
same trace as the GPU/TPU path. The verifier sees identical ASTs
either way.
"""
try:
    import jax
    import jax.numpy as jnp
    import jax.experimental.pallas as pl
    import jax.experimental.pallas.tpu as pltpu
except ImportError:
    raise ImportError(
        "Pallas support requires the jax extra: pip install stile[jax]"
    ) from None

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from ...type import (
    Type, ShapeType, DataType, Sliced, dim_size, dim_full_dim, as_int,
    simplify_dim, override_dims_in_type, TagCond, Tensor, Constant,
    type_from_binary_op,
)
from ...indexing import LoopVariable, SymbolicInt, Domain
from ...specification import parse_spec_into_type
from ...verification import verify_types_equivalent, normalize, verify_exprs_equivalent
from ...reference import run_reference, check_output_against_declaration
from .. import _core as tjax_core
from .._core import (
    TypedJaxArray, loop_var_binding, _g_active_tile_overrides,
    dtype_to_datatype, _resolve_to_runtime,
)

# Bind concrete values for ``runtime_scalar(...)``s so a ``.where("B < n")``
# predicate can materialize its mask array. Same mechanism the tiled path
# uses for ``_pid_<i>``; exposed here for callers that drive the runner
# directly. The structural proof never needs the binding — only the
# numerical ``arr`` does.
bound_runtime_scalars = loop_var_binding


@dataclass
class OutputSpec:
    """
    Declares the *expected* output of a typed Pallas kernel:

      - `spec`: either a stile-spec-language string describing the value
        the output should hold (e.g. `"2 * N"`), OR a **jax reference
        function** — a callable over the kernel's typed inputs
        (positional, in call order) returning the expected
        `TypedJaxArray` (e.g. `lambda x: x * 2`). A reference is run on
        the actual call-time inputs to extract its `ExprType`; no spec
        string needed. May be ``None`` only when a top-level
        ``reference=`` is supplied to ``typed_pallas_call`` — the
        reference then provides this output's expected expression.
      - `st`: the output's `ShapeType` (the dim signature, possibly
        sliced). Doesn't have to match `spec`'s shape verbatim — slice
        overrides happen via `verify_types_equivalent`.
      - `dt`: the output's dtype.
    """
    spec : "str | Callable | None"
    st : ShapeType
    dt : DataType | None = None


class TypedRef:
    """
    A Pallas `Ref` paired with a `Type`. `.load()` reads the entire ref
    and returns a `TypedJaxArray` carrying the same `Type` — so the
    loaded value's expression is the input's expression and downstream
    `tjax` ops compose normally.
    """
    def __init__(self, ref, type : Type):
        self.ref = ref
        self.type = type

    def load(self) -> TypedJaxArray:
        return TypedJaxArray(self.ref[...], self.type)

    def typed_value(self) -> TypedJaxArray:
        """Type-only load (``arr=None``) — for value-style reference bodies
        that need the input's type without emitting a runtime ref read."""
        return TypedJaxArray(None, self.type)

    def at(self, *idx) -> "TypedRef":
        """
        Index the underlying ref (``self.ref.at[*idx]``) and drop the
        consumed leading dims from the type's ``ShapeType``. The
        ``ExprType`` stays the original leaf — under
        ``tpl.fori_loop``'s body-equivalence trace there is one
        ``.at(layer)`` call, so "the layer-``k`` slice" is the same leaf
        for the proof. The result is a ``TypedRef`` you can
        ``.copy_from``/``.load()`` or hand to a raw block via ``.ref``.

        Each ``idx`` entry consumes one leading dim (an int / scalar
        tracer / ``SymbolicInt``) or keeps it (a ``pl.ds`` / ``slice``).
        A ``SymbolicInt`` with no ``runtime_value`` (type-only reference
        body) skips the ref index — the result's ``.ref`` is ``None``
        and only ``.typed_value()`` is valid.
        """
        rt_idx = tuple(
            i.runtime_value if isinstance(i, SymbolicInt) else i for i in idx
        )
        n_drop = sum(
            0 if isinstance(i, (slice,)) or hasattr(i, "start") else 1 for i in rt_idx
        )
        new_type = Type(self.type.st[n_drop:], self.type.et, self.type.dt)
        if any(i is None for i in rt_idx):
            return TypedRef(None, new_type)
        return TypedRef(
            self.ref.at[rt_idx if len(rt_idx) > 1 else rt_idx[0]], new_type
        )


@dataclass
class ScratchSpec:
    """
    Declares a real Pallas VMEM scratch ref for ``typed_pallas_call``.

    A ``scratch=`` entry that is a :class:`ScratchSpec` (rather than a
    bare name) is allocated via ``pl.pallas_call(scratch_shapes=...)`` —
    the kernel receives a :class:`TypedScratchRef` wrapping the real
    Pallas ref, so ``.store(x)``/``.load()`` are real VMEM writes/reads
    and ``.copy_from(hbm_ref)`` lowers to a real DMA. A bare-name entry
    keeps the Python-level-holder behavior (no allocation; the type
    flows through but the storage indirection is elided).
    """
    name : str
    shape : tuple
    dtype : "jnp.dtype" = jnp.float32


class TypedScratchRef:
    """
    A VMEM-style scratch ref — a typed mutable cell with no spec.

    Models the Pallas pattern of staging an intermediate through a VMEM
    scratch buffer: DMA a weight in, compute on it, stash an
    intermediate, read it back later. Unlike :class:`TypedOutputRef`,
    :meth:`store` does *not* verify — it records the stored value's
    :class:`Type` so a subsequent :meth:`load` returns a
    :class:`TypedJaxArray` carrying the same expression. The verifier
    sees scratch as a no-op rename; only the final ``.assign()`` on an
    output ref is checked against the reference.

    With a real Pallas ref (declared via :class:`ScratchSpec`),
    ``.store(x)`` writes ``ref[...] = x.arr`` and ``.load()`` reads
    ``ref[...]``; ``.copy_from(typed_input_ref)`` does
    ``ref[...] = src_ref[...]`` — Mosaic lowers that as an HBM→VMEM
    DMA. Without a real ref (bare-name scratch), the same calls are a
    Python-level rename: the held ``TypedJaxArray.arr`` is a JAX tracer
    and the storage indirection is elided. The type-level result is
    identical either way.
    """
    def __init__(
        self, name : "str | None" = None, ref=None,
        *, slot : "_PartSlot | None" = None, parent=None,
    ):
        self._type : "Type | None" = None
        self._name = name
        self._ref = ref
        self._held_arr = None
        # ``slot``/``parent`` are set when this ref is a view into an
        # ``UntypedScratch`` block; ``load()`` propagates them onto the
        # returned ``TypedJaxArray`` so list-form ``einsum`` can coalesce.
        self._slot = slot
        self._parent = parent

    def store(self, value : TypedJaxArray) -> None:
        if not isinstance(value, TypedJaxArray):
            raise TypeError(
                f"TypedScratchRef.store expects a TypedJaxArray; got "
                f"{type(value).__name__}"
            )
        self._type = value.type
        if self._ref is not None:
            self._ref[...] = value.arr
        else:
            self._held_arr = value.arr

    def load(self) -> TypedJaxArray:
        if self._type is None:
            who = f" {self._name!r}" if self._name else ""
            raise ValueError(f"scratch ref{who} loaded before any store")
        arr = self._ref[...] if self._ref is not None else self._held_arr
        return TypedJaxArray(
            arr, self._type, slot=self._slot, parent=self._parent,
        )

    def copy_from(self, src : "TypedJaxArray | TypedRef") -> None:
        """
        An HBM→VMEM DMA. With a real scratch ref and a :class:`TypedRef`
        source, emits ``pltpu.sync_copy(src_ref, self_ref)`` — Mosaic's
        only sanctioned way to read an ``ANY``-memory-space ref. With a
        Python-holder scratch the source must be VMEM/SMEM-loadable. The
        verifier sees the DMA as the identity: the scratch's type becomes
        ``src``'s.
        """
        if isinstance(src, TypedRef):
            self._type = src.type
            if self._ref is not None:
                pltpu.sync_copy(src.ref, self._ref)
            else:
                self._held_arr = src.ref[...]
        else:
            self.store(src)

    def axiom(self, declared : TypedJaxArray) -> None:
        """
        Declare — without proof — that this scratch ref now holds
        ``declared``'s expression. Use after a raw-Pallas block (a
        collective, a hand-tuned DMA pipeline) wrote the ref via
        ``self._ref`` directly: the verifier didn't see that write, so
        ``.axiom(x)`` records ``x.type`` for the next ``.load()``. Does
        *not* touch the ref (the raw block already did); contrast with
        :meth:`store`, which both writes and types.

        Typical pattern — a cross-device collective writes the ref via raw
        Pallas, then its typed effect is declared::

            raw_all_reduce(ar_in.ref, ar_out.ref, ...)
            ar_out.axiom(ar_in.load().sum(dev))
        """
        if not isinstance(declared, TypedJaxArray):
            raise TypeError(
                f"axiom expects a TypedJaxArray; got {type(declared).__name__}"
            )
        self._type = declared.type
        if self._ref is None:
            self._held_arr = declared.arr

    @property
    def ref(self):
        """The underlying Pallas ref (real-ref scratch only) — for handing
        to a raw-Pallas block that writes it directly. Pair with
        :meth:`axiom` to declare the typed effect afterwards."""
        if self._ref is None:
            raise ValueError(
                f"scratch ref {self._name!r} is a Python-level holder; "
                f"declare via ScratchSpec/UntypedScratchSpec for a real ref"
            )
        return self._ref

    def copy_from_async(self, src : "TypedRef", sem) -> "_AsyncDma":
        """
        Start an async HBM→VMEM DMA and return a handle whose
        :meth:`_AsyncDma.wait` blocks on completion. The scratch's type
        is recorded immediately (the verifier sees the DMA as identity,
        so a :meth:`load` after :meth:`_AsyncDma.wait` carries ``src``'s
        type); the runtime emits ``pltpu.make_async_copy(src, self,
        sem)`` so compute between ``start`` and ``wait`` overlaps the
        transfer. Real-ref scratch only; ``sem`` is a raw Pallas DMA
        semaphore (declared via ``scratch=`` as ``pltpu.SemaphoreType.DMA``
        passthrough — semaphores carry no type).
        """
        if self._ref is None:
            raise ValueError(
                "copy_from_async requires a real scratch ref (declare via "
                "ScratchSpec/UntypedScratchSpec)"
            )
        self._type = src.type
        dma = pltpu.make_async_copy(src.ref, self._ref, sem)
        dma.start()
        return _AsyncDma(dma)


@dataclass
class _AsyncDma:
    """Handle from :meth:`TypedScratchRef.copy_from_async`."""
    _dma : "object"

    def wait(self) -> None:
        self._dma.wait()


def _domain_runtime_bool(dom : Domain):
    """Evaluate a single-conjunct ``Domain`` to a runtime jax bool by
    resolving each atom via its ``runtime_value`` (the same path
    ``_resolve_to_runtime`` uses for slice offsets)."""
    result = None
    for conj in dom.disjuncts:
        c_bool = None
        for c in conj:
            v = _resolve_to_runtime(c.expr, tjax_core._loop_var_resolver) >= 0
            c_bool = v if c_bool is None else (c_bool & v)
        c_bool = True if c_bool is None else c_bool
        result = c_bool if result is None else (result | c_bool)
    return False if result is None else result


def _cond_type(then_ty : Type, domain, else_ty : Type) -> Type:
    """``mask(domain)·then + (1−mask(domain))·else`` at the ET level —
    the normalizer already handles ``TagCond``-tagged mask tensors and
    distributive products, so two ``cond``/``when`` calls with the same
    domain and operands normalize equal."""
    one = Constant(1.0)
    zero = Constant(0.0)
    mask = Tensor(dims=(), tag=TagCond(domain=domain, if_true=one, if_false=zero), name="_mask")
    not_mask = Tensor(dims=(), tag=TagCond(domain=domain, if_true=zero, if_false=one), name="_mask")
    m_then = type_from_binary_op(Type((), mask), then_ty, "*")
    m_else = type_from_binary_op(Type((), not_mask), else_ty, "*")
    return type_from_binary_op(m_then, m_else, "+")


def cond(pred : Domain, then_v : TypedJaxArray, else_v : TypedJaxArray) -> TypedJaxArray:
    """
    Value-level conditional: ``then_v`` where ``pred`` holds, else
    ``else_v``. The ET is the multiplicative form
    ``mask(pred)·then + (1−mask(pred))·else`` — the same form
    :func:`when` wraps a tagged carry's type in, so a reference body's
    ``tpl.cond(k > 0, x, x_prev)`` normalizes equal to a kernel body's
    ``carry.store(x)`` inside ``@tpl.when(k > 0, tags=(carry,))``.
    Runtime: the domain is evaluated via ``runtime_value`` and lowered
    to ``jnp.where``; with no runtime binding (type-only reference), the
    arr stays ``None``.
    """
    new_type = _cond_type(then_v.type, pred, else_v.type)
    if then_v.arr is None or else_v.arr is None:
        return TypedJaxArray(None, new_type)
    return TypedJaxArray(
        jnp.where(_domain_runtime_bool(pred), then_v.arr, else_v.arr), new_type,
    )


def fori_loop(
    lower, upper, body, *, name : str = "k",
    reference_body=None, carries : "tuple[TypedScratchRef, ...]" = (),
):
    """
    Typed ``lax.fori_loop`` over a ref-mutating body.

    ``body`` receives a ``SymbolicInt(name, runtime_value=tracer)`` —
    so ``k > 0`` is a stile ``Domain`` (for :func:`when`'s
    ``TagCond``), ``ref.at(k)`` uses the runtime tracer, and ``k * BN``
    is an ``AffineExpr`` for slice bounds — and mutates whichever
    :class:`TypedScratchRef` it closes over via
    ``.store``/``.copy_from``/``.axiom``.

    **Body equivalence** (``reference_body=`` + ``carries=``): each
    carry is rebound to a fresh ``Tensor(name=f"{ref}@{name}")`` leaf
    before the body's single ``lax.fori_loop`` trace; after the trace,
    ``reference_body(k, *fresh_leaf_values) -> tuple[TypedJaxArray]`` is
    evaluated type-only and each carry's body-output ET is checked equal
    to the reference's. Post-loop, each carry's type is an opaque
    ``_fori_<hash>_<i>`` leaf — same hash both loops produce — so the
    verifier discharges "body ≡ reference_body ⇒ final ≡ final" by
    induction, trip-count-independent.

    Without ``reference_body``: the body is traced (typed stores flow);
    post-loop carry types are "after one iteration from init" —
    ``.axiom(...)`` them if read after the loop.

    Runtime: emits ``lax.fori_loop`` with a dummy int carry.
    """

    fresh_values : list[TypedJaxArray] = []
    if reference_body is not None:
        for c in carries:
            full = tuple(dim_full_dim(d) for d in c._type.st)
            leaf = Type(c._type.st, Tensor(dims=full, name=f"{c._name}@{name}"), c._type.dt)
            c._type = leaf
            fresh_values.append(TypedJaxArray(None, leaf))

    jax.lax.fori_loop(
        lower, upper,
        lambda k, c: (body(SymbolicInt(name, runtime_value=k)), c)[1],
        0,
    )

    if reference_body is not None:
        body_ets = tuple(c._type.et for c in carries)
        ref_out = reference_body(SymbolicInt(name), *fresh_values)
        ref_out = ref_out if isinstance(ref_out, tuple) else (ref_out,)
        for i, (be, ro) in enumerate(zip(body_ets, ref_out)):
            if not verify_exprs_equivalent(be, ro.type.et):
                raise AssertionError(
                    f"tpl.fori_loop body does not match reference_body at "
                    f"carry {carries[i]._name!r}."
                )
        sig = repr((lower, upper, tuple(repr(normalize(e)) for e in body_ets)))
        h = hashlib.sha256(sig.encode()).hexdigest()[:16]
        for i, c in enumerate(carries):
            full = tuple(dim_full_dim(d) for d in c._type.st)
            c._type = Type(c._type.st, Tensor(dims=full, name=f"_fori_{h}_{i}"), None)


def when(pred, tags : "tuple[TypedScratchRef, ...]" = ()):
    """
    Typed ``pl.when`` — runtime-gates the body and ``TagCond``-wraps the
    types of the listed carry refs.

    ``pred`` is either a stile :class:`Domain` (from ``k > 0`` where
    ``k`` is the ``SymbolicInt`` :func:`fori_loop` hands the body — the
    runtime bool is derived by evaluating the domain's affine
    constraints via the atom's ``runtime_value``) or a raw runtime bool
    (no type effect). For each ref in ``tags``, the pre-body type is
    snapshotted; on exit the ref's type becomes
    ``TagCond(pred, body_type, pre_type)`` — so a reference that writes
    ``x.where("k > 0")`` normalizes equal.

    Decorator over a nullary function, same as ``pl.when``::

        @tpl.when(k > 0, tags=(carry_a, carry_b))
        def _():
            carry_a.store(...); carry_b.store(...)
    """
    if isinstance(pred, Domain):
        rt_cond, dom = _domain_runtime_bool(pred), pred
    else:
        rt_cond, dom = pred, None

    def decorator(fn):
        pre = [c._type for c in tags]
        pl.when(rt_cond)(fn)
        if dom is not None:
            for c, old in zip(tags, pre):
                if c._type is not old:
                    c._type = _cond_type(c._type, dom, old)
        return fn

    return decorator


@dataclass
class _PartSlot:
    """One contiguous view's placement inside its parent untyped buffer."""
    parent : str
    offset : int
    extent : int


def _normalize_parts(name : str, parts):
    if isinstance(parts, int):
        return [(f"{name}_{i}", 1) for i in range(parts)]
    return [
        p if isinstance(p, tuple) else (f"{name}_{i}", int(p))
        for i, p in enumerate(parts)
    ]


@dataclass
class UntypedScratchSpec:
    """
    Declares a real Pallas VMEM block carved into contiguous views.

    The block's shape is ``(sum(extents), *trailing)``; each view is the
    ``pl.ds(offset, extent)`` slice of axis 0. The kernel receives an
    :class:`UntypedScratch` whose views wrap real ``TransformedRef``s
    into the one block — ``.store``/``.load``/``.copy_from`` are real
    VMEM writes/reads/DMAs, and a list-form ``tjax.einsum`` over the
    views reads the parent buffer directly (no ``concatenate``).
    """
    name : str
    parts : "int | list"
    trailing : tuple
    dtype : "jnp.dtype" = jnp.float32

    @property
    def shape(self):
        norm = _normalize_parts(self.name, self.parts)
        return (sum(int(e) for _, e in norm), *self.trailing)


class UntypedScratch:
    """
    A flat VMEM allocation carved into a contiguous list of typed views.

    The parent buffer carries no type — only a name and a layout. Each
    view is a :class:`TypedScratchRef` (typed by what is
    ``copy_from``'d/``store``'d into it) plus a :class:`_PartSlot`
    recording its offset within the parent. With a real parent ref
    (declared via :class:`UntypedScratchSpec`) each view's ``_ref`` is
    the ``parent.at[pl.ds(offset, extent)]`` slice — a real Pallas
    ``TransformedRef`` — so stores/loads/DMAs hit the one physical
    block, and list-form ``tjax.einsum`` reads ``parent[...]`` directly
    (no ``concatenate``). Without a real ref (the in-body
    :func:`untyped_scratch` form) the views are Python-level holders and
    coalescing concatenates; the type-level result is identical.

    Index by position or by part name.
    """
    def __init__(self, name : str, parts, ref=None):
        self._name = name
        self._ref = ref
        self._views : list[TypedScratchRef] = []
        self._by_name : dict[str, TypedScratchRef] = {}
        offset = 0
        for pname, extent in _normalize_parts(name, parts):
            extent = int(extent)
            view_ref = ref.at[pl.ds(offset, extent)] if ref is not None else None
            v = TypedScratchRef(
                f"{name}.{pname}",
                ref=view_ref,
                slot=_PartSlot(parent=name, offset=offset, extent=extent),
                parent=self,
            )
            offset += extent
            self._views.append(v)
            self._by_name[pname] = v
        self._total = offset

    def __iter__(self):
        return iter(self._views)

    def __len__(self):
        return len(self._views)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_name[key]
        return self._views[key]

    def loads(self) -> list[TypedJaxArray]:
        """``[v.load() for v in self]`` — the list ``tjax.einsum`` accepts."""
        return [v.load() for v in self._views]

    def parent_arr(self):
        """The whole-block array (real-ref only) — what a coalesced matmul
        reads instead of concatenating per-view loads."""
        return None if self._ref is None else self._ref[...]


def untyped_scratch_over(name : str, ref, parts) -> UntypedScratch:
    """An :class:`UntypedScratch` whose parent ref is an *existing* raw
    Pallas ref (e.g., one allocated via ``pl.pallas_call(scratch_shapes=)``
    by an existing kernel you're porting). Same as
    declaring it via :class:`UntypedScratchSpec`, but for the drop-in
    case where ``pl.pallas_call`` already allocated the buffer."""
    return UntypedScratch(name, parts, ref=ref)


def untyped_scratch(name : str, parts) -> UntypedScratch:
    """
    A contiguous list of :class:`TypedScratchRef` views into one untyped
    parent buffer, **Python-level** (no VMEM allocation). For a real
    VMEM block, declare an :class:`UntypedScratchSpec` in
    ``typed_pallas_call(scratch=...)`` instead — the kernel then
    receives an :class:`UntypedScratch` whose views are real ref slices.
    """
    return UntypedScratch(name, parts)


def _view_slot(value : TypedJaxArray) -> "_PartSlot | None":
    """The originating scratch view's placement, if ``value`` came from one."""
    return value._slot


class TypedOutputRef(TypedRef):
    """
    The output ref of a typed Pallas kernel. `.assign(value)` runs the
    verifier against the expected `Type` and stores; if the `value`'s
    expression doesn't normalize to the expected one, raises before the
    store. The expected `Type` is resolved up front by `typed_pallas_call`
    (from a spec string or a reference function), so this just compares.
    """
    def __init__(self, ref, type : Type, output_spec : OutputSpec):
        super().__init__(ref, type)
        self.output_spec = output_spec

    def assign(self, value : TypedJaxArray):
        if not verify_types_equivalent(self.type, value.type):
            raise ValueError(
                "Pallas output does not match spec! "
                f"Expected: {self.type.et}, actual: {value.type.et}"
            )
        self.ref[...] = value.arr


def _resolve_spec_et(out_type : OutputSpec, typed_inputs):
    """
    Resolve an `OutputSpec` to the expected output `ExprType`.

    For a spec string, parse it. For a reference function, run it on the
    call-time inputs (re-wrapped so their stile `Type` carries the real
    array dtype, making the reference's output dtype checkable) and verify
    the reference's output ShapeType / dtype against the declaration before
    returning its ExprType.
    """
    if isinstance(out_type.spec, str):
        return parse_spec_into_type(out_type.spec).et
    ref_inputs = [
        TypedJaxArray(
            ti.arr,
            Type(ti.type.st, ti.type.et, dtype_to_datatype(ti.arr.dtype)),
        )
        for ti in typed_inputs
    ]
    outs = run_reference(out_type.spec, ref_inputs)
    if len(outs) != 1:
        raise ValueError(
            "typed_pallas_call: a Pallas reference must return a single "
            f"typed value (single-output kernel); got {len(outs)}."
        )
    ref_type = outs[0].type
    check_output_against_declaration(
        ref_type, out_type.st, dtype_to_datatype(out_type.dt),
        label="typed_pallas_call reference",
    )
    return ref_type.et


def _block_sliced_dim(parent_dim, block_idx, block_size):
    """
    Combine a parent dim with a block-relative position and size to
    produce the sliced dim the in-kernel ref actually exposes:
    `[block_idx * block_size, block_idx * block_size + block_size)`.
    `block_idx` is whatever `BlockSpec.index_map(*pids)` returned for
    this axis — a plain `int` for static positions, an `AffineExpr`
    when `program_id` LoopVariables flowed through. `simplify_dim`
    folds the trivial `Sliced(D, 0, D.size) → D` case.
    """
    abs_start = block_idx * block_size
    abs_end = abs_start + block_size
    return simplify_dim(Sliced(parent_dim, abs_start, abs_end))


def _derive_sliced_st(parent_st, block_spec, pids):
    """
    Build the ShapeType the kernel-side sees for a ref governed by
    `block_spec` under `pids` (one `LoopVariable` per grid axis).
    """
    block_indices = block_spec.index_map(*pids)
    block_shape = block_spec.block_shape
    sliced = []
    for parent_dim, idx, sz in zip(parent_st, block_indices, block_shape):
        sliced.append(_block_sliced_dim(parent_dim, idx, sz))
    return tuple(sliced)


def _normalize_out_types(out_type):
    """Return ``(tuple_of_OutputSpec, was_single)`` for either form."""
    if isinstance(out_type, OutputSpec):
        return (out_type,), True
    return tuple(out_type), False


def _out_struct(spec : OutputSpec):
    dims = tuple(as_int(dim_size(d)) for d in spec.st)
    return jax.ShapeDtypeStruct(dims, spec.dt if spec.dt is not None else jnp.float32)


def _resolve_spec_ets(out_types, typed_inputs, reference):
    """
    Resolve each output's expected ``ExprType``.

    With a top-level ``reference`` (a tjax callable over the typed
    inputs returning one ``TypedJaxArray`` per output), run it once and
    take each return value's ``.type.et`` — checking shape/dtype against
    the declared ``OutputSpec``. Otherwise fall back to the per-output
    ``spec`` (string or callable) via ``_resolve_spec_et``.
    """
    if reference is None:
        for ot in out_types:
            if ot.spec is None:
                raise ValueError(
                    "OutputSpec.spec is None but typed_pallas_call has no "
                    "reference= — supply one or the other."
                )
        return [_resolve_spec_et(ot, typed_inputs) for ot in out_types]
    for ot in out_types:
        if ot.spec is not None:
            raise ValueError(
                "typed_pallas_call: pass either reference= or per-output "
                "OutputSpec.spec, not both."
            )
    ref_inputs = [
        TypedJaxArray(
            ti.arr,
            Type(ti.type.st, ti.type.et, dtype_to_datatype(ti.arr.dtype)),
        )
        for ti in typed_inputs
    ]
    outs = run_reference(reference, ref_inputs)
    if len(outs) != len(out_types):
        raise ValueError(
            f"typed_pallas_call reference returned {len(outs)} value(s) but "
            f"out_type declares {len(out_types)}."
        )
    ets = []
    for out, ot in zip(outs, out_types):
        check_output_against_declaration(
            out.type, ot.st, dtype_to_datatype(ot.dt),
            label="typed_pallas_call reference",
        )
        ets.append(out.type.et)
    return ets


def typed_pallas_call(
    kernel_fn,
    out_type : "OutputSpec | tuple[OutputSpec, ...] | list[OutputSpec]",
    *,
    reference : "Callable | None" = None,
    grid=None,
    in_specs=None,
    out_specs=None,
    scratch : "int | tuple[str | ScratchSpec | UntypedScratchSpec | Any, ...] | list[str | ScratchSpec | UntypedScratchSpec | Any]" = 0,
    interpret : bool = True,
    compiler_params=None,
):
    """
    Wrap a Pallas kernel function with stile typing. The user's
    `kernel_fn` takes one ref per input followed by one
    `TypedOutputRef` per output. The returned callable takes typed (or
    raw) inputs and produces a `TypedJaxArray` (or tuple thereof).

    **Multi-output**: pass a tuple/list of `OutputSpec` for `out_type`.
    The kernel receives that many `TypedOutputRef`s; the runner returns
    that many `TypedJaxArray`s. A single `OutputSpec` keeps the
    single-value calling convention.

    **Untyped passthrough**: an input that is *not* a `TypedJaxArray`
    is forwarded as a raw `jax.Array` and the kernel receives the bare
    Pallas ref for it — for index tables, peer/slot tables, and other
    operands the verifier should not see.

    **Reference verification**: instead of per-output spec strings,
    pass ``reference=`` — a tjax callable over the typed inputs (in
    call order, untyped inputs skipped) that returns one
    ``TypedJaxArray`` per output. The kernel is then proven equivalent
    to whatever the reference computes; no spec language needed.

    For tiled kernels, pass `grid` (a tuple of grid axis sizes) and
    `in_specs` / `out_specs` (raw `pl.BlockSpec`s, one per input /
    output). Inside the kernel, each ref is already block-sliced by
    Pallas; stile derives the ref's `Type` by feeding `LoopVariable`s
    into the BlockSpec's `index_map` so the type's ShapeType reflects
    the symbolic slice. The output ref's expected type is the spec
    restricted to the tile via `override_dims_in_type` — so per-block
    `assign(...)` certifies "this block matches the spec's tile."

    **VMEM scratch**: pass ``scratch=`` — either an int (number of
    anonymous scratch refs) or a tuple of names. The kernel receives
    that many :class:`TypedScratchRef` objects after the output refs.
    Scratch refs are unverified mutable cells: ``.store(x)`` records
    ``x``'s Type, ``.load()`` returns it. Use them to mirror a Pallas
    kernel's VMEM staging (DMA-in, carry across phases) without the
    verifier seeing a cut.

    `interpret=True` (default) runs the kernel on CPU via Pallas's
    interpreter. Same trace as the GPU/TPU path, so the verifier sees
    identical ASTs — let the dev loop be local, the perf loop be
    remote.
    """
    out_types, single_out = _normalize_out_types(out_type)
    out_structs = tuple(_out_struct(s) for s in out_types)
    if isinstance(scratch, int):
        scratch = tuple(f"scratch_{i}" for i in range(scratch))
    # Split scratch into Pallas-allocated (ScratchSpec/UntypedScratchSpec →
    # pltpu.VMEM; raw pltpu.SemaphoreType / pltpu.VMEM passthrough) and
    # Python-level holders (bare names — no allocation). The kernel receives
    # them in declaration order; the Pallas-allocated refs arrive after the
    # output refs.
    scratch_decls = tuple(scratch)
    pl_scratch_shapes : list = []
    for s in scratch_decls:
        if isinstance(s, (ScratchSpec, UntypedScratchSpec)):
            pl_scratch_shapes.append(pltpu.VMEM(s.shape, s.dtype))
        elif not isinstance(s, str):
            pl_scratch_shapes.append(s)

    tiled = grid is not None
    # ``in_specs`` is dual-purpose: with ``grid=`` it carries per-axis
    # block slicing (the tiled path below); without, it's a
    # memory-space-only passthrough (e.g. ``BlockSpec(memory_space=pl.ANY)``
    # to keep big weights in HBM) and the input's Type is unchanged.
    nontiled_in_specs = in_specs if (in_specs is not None and not tiled) else None

    def runner(*inputs):
        n_in = len(inputs)
        # Untyped inputs pass through to Pallas unchanged; typed inputs
        # contribute their `.arr`.
        raw_inputs = [
            ti.arr if isinstance(ti, TypedJaxArray) else ti for ti in inputs
        ]
        typed_only = [ti for ti in inputs if isinstance(ti, TypedJaxArray)]
        # Resolve each output's expected ExprType once — from a top-level
        # reference, a per-output spec string, or a per-output reference.
        # The reference sees only typed inputs, in call order.
        spec_ets = _resolve_spec_ets(out_types, typed_only, reference)

        def jax_kernel(*refs):
            input_refs = refs[:n_in]
            output_refs = refs[n_in : n_in + len(out_types)]
            real_scratch_refs = refs[n_in + len(out_types):]
            # Build the kernel-order scratch list, drawing real refs from
            # Pallas where declared and constructing Python-level holders
            # for bare names. Raw pltpu scratch (semaphores etc.) passes
            # through unwrapped — semaphores carry no type.
            real_iter = iter(real_scratch_refs)
            scratch_refs : list = []
            for s in scratch_decls:
                if isinstance(s, UntypedScratchSpec):
                    scratch_refs.append(
                        UntypedScratch(s.name, s.parts, ref=next(real_iter))
                    )
                elif isinstance(s, ScratchSpec):
                    scratch_refs.append(TypedScratchRef(s.name, ref=next(real_iter)))
                elif isinstance(s, str):
                    scratch_refs.append(TypedScratchRef(s))
                else:
                    scratch_refs.append(next(real_iter))
            if tiled:
                # `tiled` implies these were all supplied together.
                assert grid is not None and in_specs is not None and out_specs is not None
                pids = tuple(
                    LoopVariable(f"_pid_{i}") for i in range(len(grid))
                )
                # Bind each `_pid_<i>` LoopVariable to its corresponding
                # `pl.program_id(i)` jax value so symbolic slice offsets
                # used by `tjax.mask` (or any other in-kernel runtime
                # construction) can be evaluated at trace time.
                pid_runtime = {
                    f"_pid_{i}": pl.program_id(i) for i in range(len(grid))
                }
                wrapped_inputs = []
                for ref, ti, spec in zip(input_refs, inputs, in_specs):
                    if not isinstance(ti, TypedJaxArray):
                        wrapped_inputs.append(ref)
                        continue
                    sliced_st = _derive_sliced_st(ti.type.st, spec, pids)
                    wrapped_inputs.append(
                        TypedRef(ref, Type(sliced_st, ti.type.et, ti.type.dt))
                    )
                # Outputs: each spec describes the FULL output; restrict
                # it to the tile via override_dims_in_type so per-block
                # assign certifies the tile, not the global tensor.
                out_specs_seq = (
                    out_specs if isinstance(out_specs, (list, tuple)) else (out_specs,)
                )
                wrapped_outputs = []
                tile_overrides : list = []
                for ref, ot, et, pl_spec in zip(
                    output_refs, out_types, spec_ets, out_specs_seq,
                ):
                    sliced_st = _derive_sliced_st(ot.st, pl_spec, pids)
                    full_spec_type = Type(ot.st, et, ot.dt)
                    tile_spec = override_dims_in_type(full_spec_type, *sliced_st)
                    wrapped_outputs.append(TypedOutputRef(
                        ref, Type(sliced_st, tile_spec.et, ot.dt), ot,
                    ))
                    tile_overrides.extend(
                        d for d in sliced_st if isinstance(d, Sliced)
                    )
                # The tile context: every Sliced dim that the kernel
                # body operates on. Inner `tjax.fori_loop(..., invariant=...)`
                # calls read this stack to restrict their parsed
                # invariant types to the tile, so the body's Sliced
                # types match the invariant's during verification.
                _g_active_tile_overrides.append(tuple(tile_overrides))
                try:
                    with loop_var_binding(pid_runtime):
                        kernel_fn(*wrapped_inputs, *wrapped_outputs, *scratch_refs)
                finally:
                    _g_active_tile_overrides.pop()
            else:
                wrapped_inputs = [
                    TypedRef(ref, ti.type)
                    if isinstance(ti, TypedJaxArray)
                    else ref
                    for ref, ti in zip(input_refs, inputs)
                ]
                wrapped_outputs = [
                    TypedOutputRef(ref, Type(ot.st, et, ot.dt), ot)
                    for ref, ot, et in zip(output_refs, out_types, spec_ets)
                ]
                kernel_fn(*wrapped_inputs, *wrapped_outputs, *scratch_refs)

        pallas_kwargs = {}
        if tiled:
            pallas_kwargs.update(
                grid=grid, in_specs=in_specs, out_specs=out_specs,
            )
        elif nontiled_in_specs is not None:
            pallas_kwargs['in_specs'] = nontiled_in_specs
        if pl_scratch_shapes:
            pallas_kwargs['scratch_shapes'] = pl_scratch_shapes
        if compiler_params is not None:
            pallas_kwargs['compiler_params'] = compiler_params
        result = pl.pallas_call(
            jax_kernel,
            out_shape=out_structs[0] if single_out else out_structs,
            interpret=interpret,
            **pallas_kwargs,
        )(*raw_inputs)

        # The returned TypedJaxArrays carry the resolved ExprTypes (so
        # downstream consumers see the spec / reference, not the kernel's
        # internal expression) and each OutputSpec's ShapeType.
        result_arrs = (result,) if single_out else tuple(result)
        typed_outs = tuple(
            TypedJaxArray(arr, Type(ot.st, et, ot.dt))
            for arr, ot, et in zip(result_arrs, out_types, spec_ets)
        )
        return typed_outs[0] if single_out else typed_outs

    return runner
