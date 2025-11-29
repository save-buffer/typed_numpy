import math

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from type_nodes import *

@dataclass(frozen=True)
class EclassID:
    id : int

class UnionFind:
    def __init__(self):
        self.parent : list[EclassID] = []
        self.rank : list[int] = []

    def make_class(self) -> EclassID:
        next_id = len(self.parent)
        self.parent.append(EclassID(next_id))
        self.rank.append(0)
        return self.parent[-1]

    def find(self, x : EclassID) -> EclassID:
        while self.parent[x.id] != x:
            self.parent[x.id] = self.parent[self.parent[x.id].id]
            x = self.parent[x.id]
        return x

    def union(self, x : EclassID, y : EclassID) -> EclassID:
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return x

        if self.rank[x.id] < self.rank[y.id]:
            x, y = y, x

        self.parent[y.id] = x
        if self.rank[x.id] == self.rank[y.id]:
            self.rank[x.id] += 1
        return x

class EnodeType(Enum):
    Constant = "Constant"
    Dimension = "Dimension"
    Tensor = "Tensor"
    Exp = "Exp"
    Sin = "Sin"
    Cos = "Cos"
    Sqrt = "Sqrt"
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    BinaryMax = "BinaryMax"
    Repeat = "Repeat"
    Sum = "Sum"
    Max = "Max"

def enode_type_from_unary_op(x : UnaryOpType):
    match x:
        case "exp":
            return EnodeType.Exp
        case "sin":
            return EnodeType.Sin
        case "cos":
            return EnodeType.Cos

def enode_type_from_binary_op(x : BinaryOpType):
    match x:
        case "+" | "-" | "*" | "/":
            return EnodeType(x)
        case "max":
            return EnodeType.BinaryMax

def enode_type_from_reduce_op(x : ReduceOpType):
    match x:
        case "sum":
            return EnodeType.Sum
        case "max":
            return EnodeType.Max

@dataclass(frozen=True)
class Enode:
    op : EnodeType
    args : tuple[Any, ...]

class Egraph:
    def __init__(self):
        self.uf = UnionFind()
        self.eclasses : dict[EclassID, set[Enode]] = defaultdict(set)
        self.enodes : dict[Enode, EclassID] = {}
        
        self._rules : list[Callable[[EclassID, Enode], bool]] = [
            self._commutativity,
            self._associativity,
            self._distributivity,
            self._inverse_distributivity,
            self._combine_reductions,
            self._mul_or_div_into_reduction_or_repeat,
            self._mul_or_div_out_of_reduction_or_repeat,
            self._mul_by_zero,
            self._div_by_one,
            self._sub_or_div_by_self,
            self._identity,
            self._constant_folding_unary,
            self._constant_folding_binary,
            self._decompose_exp,
        ]

    def _commutativity(self, id : EclassID, enode : Enode) -> bool:
        # a + b = b + a
        if enode.op not in (EnodeType.Add, EnodeType.Mul):
            return False
        a, b = enode.args
        swapped_id = self.add(
            Enode(
                op=enode.op,
                args=(b, a)
            )
        )
        if not self.equivalent(id, swapped_id):
            self.merge(id, swapped_id)
            return True
        return False

    def _associativity(self, id : EclassID, enode : Enode) -> bool:
        # (a + b) + c = a + (b + c)
        if enode.op not in (EnodeType.Add, EnodeType.Mul):
            return False
        left, c = enode.args
        left_enodes = self.get_enodes(left)
        for enode_left in left_enodes:
            if enode_left.op == enode.op:
                a, b = enode_left.args
                bc = self.add(
                    Enode(
                        op=enode.op,
                        args=(b, c),
                    )
                )
                reassoc = self.add(
                    Enode(
                        op=enode.op,
                        args=(a, bc),
                    )
                )
                if not self.equivalent(id, reassoc):
                    self.merge(id, reassoc)
                    return True
        return False

    def _distributivity(self, id : EclassID, enode : Enode) -> bool:
        # a * (b + c) = a * b + a * c
        if enode.op != EnodeType.Mul:
            return False
        a, bc = enode.args
        bc_enodes = self.get_enodes(bc)
        for bc_enode in bc_enodes:
            if bc_enode.op == EnodeType.Add:
                b, c = bc_enode.args
                ab = self.add(
                    Enode(
                        op=EnodeType.Mul,
                        args=(a, b),
                    )
                )
                ac = self.add(
                    Enode(
                        op=EnodeType.Mul,
                        args=(a, c),
                    )
                )
                abac = self.add(
                    Enode(
                        op=EnodeType.Add,
                        args=(ab, ac),
                    )
                )
                if not self.equivalent(id, abac):
                    self.merge(id, abac)
                    return True
        return False

    def _inverse_distributivity(self, id : EclassID, enode : Enode) -> bool:
        # a * b + a * c = a * (b + c)
        if enode.op != EnodeType.Add:
            return False
        ab, ac = enode.args
        ab_enodes = self.get_enodes(ab)
        ac_enodes = self.get_enodes(ac)
        for ab_enode in ab_enodes:
            if ab_enode.op != EnodeType.Mul:
                continue
            for ac_enode in ac_enodes:
                if ac_enode.op != EnodeType.Mul:
                    continue
                la, b = ab_enode.args
                ra, c = ab_enode.args
                if not self.equivalent(la, ra):
                    continue
                bc = self.add(
                    Enode(
                        op=EnodeType.Add,
                        args=(b, c),
                    )
                )
                abc = self.add(
                    Enode(
                        op=EnodeType.Mul,
                        args=(la, bc),
                    )
                )
                if not self.equivalent(id, abc):
                    self.merge(id, abc)
                    return True
        return False

    def _combine_reductions(self, id : EclassID, enode : Enode) -> bool:
        # sum[x:y] + sum[y:z] = sum(x:z) or binary_max(max[x:y], max[y:z]) = max[x:z]
        match enode.op:
            case EnodeType.Add:
                expected_child_op = EnodeType.Sum
            case EnodeType.BinaryMax:
                expected_child_op = EnodeType.Max
            case _:
                return False

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != expected_child_op:
                continue
            lhs_dim, lhs_child = lhs_enode.args

            for rhs_enode in rhs_enodes:
                if rhs_enode.op != expected_child_op:
                    continue
                rhs_dim, rhs_child = rhs_enode.args
                if (
                        self.equivalent(lhs_child, rhs_child)
                        and dim_full_dim(lhs_dim) == dim_full_dim(rhs_dim)
                        and dim_end(lhs_dim) == dim_start(rhs_dim)
                ):
                    combined_dim = simplify_dim(
                        Sliced(
                            dim_full_dim(lhs_dim),
                            dim_start(lhs_dim),
                            dim_end(rhs_dim),
                        )
                    )
                    combined = self.add(
                        Enode(
                            op=expected_child_op,
                            args=(combined_dim, lhs_child),
                        )
                    )
                    if not self.equivalent(id, combined):
                        self.merge(id, combined)
                        return True
        return False

    def _mul_or_div_into_reduction_or_repeat(self, id : EclassID, enode : Enode) -> bool:
        # sum(x / c) = sum(x) / c, or max(x / c) = max(x) / c if c >= 0
        if enode.op not in (EnodeType.Mul, EnodeType.Div):
            return False
        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op not in (EnodeType.Repeat, EnodeType.Sum, EnodeType.Max):
                continue

            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Constant:
                    continue

                val, = rhs_enode.args
                if lhs_enode.op == EnodeType.Max and val < 0:
                    continue

                lhs_dim, lhs_child = lhs_enode.args
                new_inner = self.add(
                    Enode(
                        op=enode.op,
                        args=(lhs_child, rhs),
                    )
                )
                new_outer = self.add(
                    Enode(
                        op=lhs_enode.op,
                        args=(lhs_dim, new_inner),
                    )
                )
                if not self.equivalent(id, new_outer):
                    self.merge(id, new_outer)
                    return True
        return False

    def _mul_or_div_out_of_reduction_or_repeat(self, id : EclassID, enode : Enode) -> bool:
        # sum(x / c) = sum(x) / c, or max(x / c) = max(x) / c if c >= 0
        if enode.op not in (EnodeType.Repeat, EnodeType.Sum, EnodeType.Max):
            return False

        dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op not in (EnodeType.Mul, EnodeType.Div):
                continue
            child_lhs, child_rhs = child_enode.args
            child_rhs_enodes = self.get_enodes(child_rhs)
            for child_rhs_enode in child_rhs_enodes:
                if child_rhs_enode.op != EnodeType.Constant:
                    continue
                val, = child_rhs_enode.args
                if enode.op == EnodeType.Max and val < 0:
                    continue

                new_inner = self.add(
                    Enode(
                        op=enode.op,
                        args=(dim, child_lhs),
                    )
                )
                new_outer = self.add(
                    Enode(
                        op=child_enode.op,
                        args=(new_inner, child_rhs),
                    )
                )
                if not self.equivalent(id, new_outer):
                    self.merge(id, new_outer)
                    return True
        return False

    def _mul_by_zero(self, id : EclassID, enode : Enode) -> bool:
        if enode.op != EnodeType.Mul:
            return False

        lhs, rhs = enode.args
        rhs_enodes = self.get_enodes(rhs)
        for rhs_enode in rhs_enodes:
            if rhs_enode.op == EnodeType.Constant:
                break
        if rhs_enode != EnodeType.Constant:
            return False
        val, = rhs_enode.args
        if val != 0:
            return False
        if not self.equivalent(id, rhs):
            self.merge(id, rhs)
            return True
        return False

    def _div_by_one(self, id : EclassID, enode : Enode) -> bool:
        if enode.op != EnodeType.Div:
            return False

        lhs, rhs = enode.args
        rhs_enodes = self.get_enodes(rhs)
        for rhs_enode in rhs_enodes:
            if rhs_enode.op == EnodeType.Constant:
                break
        if rhs_enode != EnodeType.Constant:
            return False
        val, = rhs_enode.args
        if val != 1:
            return False
        if not self.equivalent(id, rhs):
            self.merge(id, rhs)
            return True
        return False

    def _sub_or_div_by_self(self, id : EclassID, enode : Enode) -> bool:
        if enode.op not in (EnodeType.Sub, EnodeType.Div):
            return False
        lhs, rhs = enode.args
        if not self.equivalent(lhs, rhs):
            return False

        match enode.op:
            case EnodeType.Sub:
                new_val = 0.0
            case EnodeType.Div:
                new_val = 1.0
        new_constant = self.add(
            Enode(
                EnodeType.Constant,
                args=(new_val,),
            )
        )
        if not self.equivalent(id, new_constant):
            self.merge(id, new_constant)
            return True
        return False

        
    def _identity(self, id : EclassID, enode : Enode) -> bool:
        if enode.op not in (EnodeType.Add, EnodeType.Sub, EnodeType.Mul, EnodeType.Div):
            return False
        
        lhs, rhs = enode.args
        rhs_enodes = self.get_enodes(rhs)
        for rhs_enode in rhs_enodes:
            if rhs_enode.op == EnodeType.Constant:
                break
        if rhs_enode.op != EnodeType.Constant:
            return False

        val, = rhs_enode.args
        match enode.op:
            case EnodeType.Add | EnodeType.Sub:
                if val != 0:
                    return False
            case EnodeType.Mul | EnodeType.Div:
                if val != 1:
                    return False
        if not self.equivalent(id, lhs):
            self.merge(id, lhs)
            return True
        return False

    def _constant_folding_unary(self, id : EclassID, enode : Enode) -> bool:
        if enode.op not in (EnodeType.Exp, EnodeType.Sin, EnodeType.Cos, EnodeType.Sqrt):
            return False

        my_enodes = self.get_enodes(id)
        for my_enode in my_enodes:
            if my_enode.op == EnodeType.Constant:
                return False

        child, = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op == EnodeType.Constant:
                break
        if child_enode.op != EnodeType.Constant:
            return False

        val, = child_enode.args
        match enode.op:
            case EnodeType.Exp:
                new_val = math.exp(val)
            case EnodeType.Sin:
                new_val = math.sin(val)
            case EnodeType.Cos:
                new_val = math.cos(val)
            case EnodeType.Sqrt:
                new_val = math.sqrt(val)
        new_const = self.add(
            Enode(
                op=EnodeType.Constant,
                args=(new_val,),
            )
        )
        if not self.equivalent(id, new_const):
            self.merge(id, new_const)
            return True
        return False


    def _constant_folding_binary(self, id : EclassID, enode : Enode) -> bool:
        if enode.op not in (EnodeType.Add, EnodeType.Sub, EnodeType.Mul, EnodeType.Div):
            return False

        # Make sure we haven't already folded this constant expression
        my_enodes = self.get_enodes(id)
        for my_enode in my_enodes:
            if my_enode.op == EnodeType.Constant:
                return False

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op == EnodeType.Constant:
                break
        if lhs_enode.op != EnodeType.Constant:
            return False

        for rhs_enode in rhs_enodes:
            if rhs_enode.op == EnodeType.Constant:
                break
        if rhs_enode.op != EnodeType.Constant:
            return False

        val_l, = lhs_enode.args
        val_r, = rhs_enode.args

        match enode.op:
            case EnodeType.Add:
                new_val = val_l + val_r
            case EnodeType.Sub:
                new_val = val_l - val_r
            case EnodeType.Mul:
                new_val = val_l * val_r
            case EnodeType.Div:
                new_val = val_l / val_r
        new_const = self.add(
            Enode(
                op=EnodeType.Constant,
                args=(new_val,),
            )
        )
        if not self.equivalent(id, new_const):
            self.merge(id, new_const)
            return True
        return False

    def _decompose_exp(self, id : EclassID, enode : Enode) -> bool:
        if enode.op != EnodeType.Exp:
            return False
        child, = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op in (EnodeType.Add, EnodeType.Sub):
                continue
        if child_enode.op not in (EnodeType.Add, EnodeType.Sub):
            return False
        
        lhs, rhs = child_enode.args
        exp_lhs = self.add(
            Enode(
                op=EnodeType.Exp,
                args=(lhs,),
            )
        )
        exp_rhs = self.add(
            Enode(
                op=EnodeType.Exp,
                args=(rhs,),
            )
        )
        match child_enode.op:
            case EnodeType.Add:
                new_op = EnodeType.Mul
            case EnodeType.Sub:
                new_op = EnodeType.Div

        combined = self.add(
            Enode(
                op=new_op,
                args=(exp_lhs, exp_rhs),
            )
        )
        if not self.equivalent(id, combined):
            self.merge(id, combined)
            return True
        return False


    def _canonicalize(self, enode : Enode) -> Enode:
        canonicalized_args = tuple(
            arg if not isinstance(arg, EclassID) else self.uf.find(arg)
            for arg in enode.args
        )

        return Enode(
            op=enode.op,
            args=canonicalized_args,
        )

    def _rebuild(self):
        old_enodes = self.enodes
        self.enodes = {}

        for enode, old_id in old_enodes.items():
            canon_enode = self._canonicalize(enode)
            canon_id = self.uf.find(old_id)

            if canon_enode in self.enodes:
                existing_id = self.enodes[canon_enode]
                self.merge(existing_id, canon_id)
            else:
                self.enodes[canon_enode] = canon_id

    def add(self, enode : Enode) -> EclassID:
        enode = self._canonicalize(enode)
        if enode in self.enodes:
            return self.uf.find(self.enodes[enode])

        new_eclass_id = self.uf.make_class()
        self.enodes[enode] = new_eclass_id
        self.eclasses[new_eclass_id].add(enode)
        return new_eclass_id

    def merge(self, x : EclassID, y : EclassID) -> EclassID:
        root_x = self.uf.find(x)
        root_y = self.uf.find(y)

        if root_x == root_y:
            return root_x

        new_root = self.uf.union(root_x, root_y)
        old_root = root_x if new_root == root_y else root_y

        self.eclasses[new_root].update(self.eclasses[old_root])
        del self.eclasses[old_root]

        self._rebuild()
        return new_root

    def get_enodes(self, x : EclassID) -> set[Enode]:
        root = self.uf.find(x)
        return self.eclasses.get(root, set())

    def equivalent(self, x : EclassID, y : EclassID) -> bool:
        return self.uf.find(x) == self.uf.find(y)

    def insert_expression(self, expr : ExprType) -> EclassID:
        match expr:
            case Constant(v):
                enode = Enode(
                    op=EnodeType.Constant,
                    args=(v,),
                )
                return self.add(enode)
            case Tensor(dims):
                enode = Enode(
                    op=EnodeType.Tensor,
                    args=(dims,),
                )
                return self.add(enode)
            case UnaryOp(op, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=enode_type_from_unary_op(op),
                    args=(child_id,),
                )
                return self.add(enode)
            case BinaryOp(op, lhs, rhs):
                lhs_id = self.insert_expression(lhs)
                rhs_id = self.insert_expression(rhs)
                enode = Enode(
                    op=enode_type_from_binary_op(op),
                    args=(lhs_id, rhs_id),
                )
                return self.add(enode)
            case Repeat(dim, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=EnodeType.Repeat,
                    args=(dim, child_id),
                )
                return self.add(enode)
            case Reduce(op, dim, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=enode_type_from_reduce_op(op),
                    args=(dim, child_id),
                )
                return self.add(enode)
        assert False, "Unrecognized expression"

    def _apply_rules_to_enode(self, eclass_id : EclassID, enode : Enode) -> bool:
        current_id = self.uf.find(eclass_id)
        for rule in self._rules:
            if rule(current_id, enode):
                return True
        return False

    def apply_rewrites(self) -> int:
        nmerges = 0
        all_enodes = []
        for eclass_id in self.eclasses.keys():
            enodes = self.get_enodes(eclass_id)
            for enode in enodes:
                all_enodes.append((eclass_id, enode))

        for eclass_id, enode in all_enodes:
            nmerges += 1 if self._apply_rules_to_enode(eclass_id, enode) else 0
        return nmerges

    def incrementally_check_equivalence(self, x : EclassID, y : EclassID, max_iters : int = 10) -> bool:
        total_merges = 0
        for i in range(max_iters):
            merges_this_iter = self.apply_rewrites()
            print(f"Iter {i} did {merges_this_iter} merges, {len(self.enodes)} enodes")            
            total_merges += merges_this_iter
            
            if self.equivalent(x, y):
                return True
        return False
        
    def equality_saturation(self, max_iters : int = 10):
        total_merges = 0
        for i in range(max_iters):
            merges_this_iter = self.apply_rewrites()
            print(f"Iter {i} did {merges_this_iter} merges, {len(self.enodes)} enodes")
            total_merges += merges_this_iter
            if merges_this_iter == 0:
                break

        print(f"Did {total_merges} merges after {i + 1} iters")

