from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class ConstraintModel:
    """
    Generic binary constraint model scaffold with optional exactly-one groups.

    - universe: security indices
    - forced: hard assignments {i:0|1}
    - groups_exactly_one: list of groups; each group is a set of indices where
      exactly one index must be 1 in any feasible vertex.

    This supports a practical first market-family model for BTC up/down style
    contracts (YES/NO pair -> exactly one true).
    """

    universe: Set[int]
    forced: Optional[Dict[int, int]] = None
    groups_exactly_one: Optional[List[Set[int]]] = None

    def unresolved_indices(self, sigma_hat: Dict[int, int]) -> List[int]:
        return sorted([i for i in self.universe if i not in sigma_hat])

    def _merged_assignments(self, sigma_hat: Dict[int, int], explicit: Optional[Dict[int, int]] = None) -> Dict[int, int]:
        merged: Dict[int, int] = {}
        for i, v in (self.forced or {}).items():
            merged[int(i)] = int(v)
        for i, v in sigma_hat.items():
            i, v = int(i), int(v)
            if i in merged and merged[i] != v:
                raise ValueError(f"conflict on {i}: forced={merged[i]} sigma={v}")
            merged[i] = v
        if explicit:
            for i, v in explicit.items():
                i, v = int(i), int(v)
                if i in merged and merged[i] != v:
                    raise ValueError(f"conflict on {i}: merged={merged[i]} explicit={v}")
                merged[i] = v
        return merged

    def is_feasible_assignment(self, i: int, value: int, sigma_hat: Dict[int, int]) -> bool:
        try:
            merged = self._merged_assignments(sigma_hat, explicit={int(i): int(value)})
        except ValueError:
            return False

        # Group consistency checks
        for g in (self.groups_exactly_one or []):
            ones = sum(1 for idx in g if merged.get(idx) == 1)
            unknown = sum(1 for idx in g if idx not in merged)

            if ones > 1:
                return False
            if unknown == 0 and ones != 1:
                return False

        return True

    def build_vertex(self, sigma_hat: Dict[int, int], explicit: Optional[Dict[int, int]] = None) -> Dict[int, int]:
        merged = self._merged_assignments(sigma_hat, explicit=explicit)

        v: Dict[int, int] = {i: int(merged.get(i, 0)) for i in self.universe}

        # Satisfy exactly-one groups deterministically
        for g in (self.groups_exactly_one or []):
            ones = [idx for idx in g if v.get(idx) == 1]
            if len(ones) > 1:
                raise ValueError(f"infeasible group (multiple ones): {g}")
            if len(ones) == 1:
                for idx in g:
                    if idx != ones[0]:
                        v[idx] = 0
                continue

            candidate = None
            for idx in sorted(g):
                if idx in merged and merged[idx] == 1:
                    candidate = idx
                    break
            if candidate is None:
                candidate = min(g)

            for idx in g:
                v[idx] = 1 if idx == candidate else 0

        if not self.is_vertex_feasible(v):
            raise ValueError("failed to build feasible vertex")

        return v

    def build_descent_vertex(self, sigma_hat: Dict[int, int], gradient: Dict[int, float]) -> Dict[int, int]:
        """
        Approximate linear minimization oracle for objective <gradient, z>.
        For each exactly-one group, choose argmin gradient within the group.
        For unconstrained binary vars, choose 1 when gradient_i < 0 else 0.
        """
        merged = self._merged_assignments(sigma_hat)
        v: Dict[int, int] = {i: int(merged.get(i, 0)) for i in self.universe}

        grouped = set()
        for g in (self.groups_exactly_one or []):
            grouped |= set(g)
            # if group already fixed to one=1 by merged assignments, honor it
            preset_ones = [idx for idx in g if idx in merged and merged[idx] == 1]
            if len(preset_ones) > 1:
                raise ValueError(f"infeasible merged assignments in group: {g}")
            if len(preset_ones) == 1:
                chosen = preset_ones[0]
            else:
                chosen = min(g, key=lambda idx: float(gradient.get(idx, 0.0)))
            for idx in g:
                v[idx] = 1 if idx == chosen else 0

        for i in self.universe:
            if i in grouped:
                continue
            if i in merged:
                v[i] = int(merged[i])
            else:
                v[i] = 1 if float(gradient.get(i, 0.0)) < 0.0 else 0

        if not self.is_vertex_feasible(v):
            raise ValueError("descent vertex infeasible")
        return v

    def is_vertex_feasible(self, vertex: Dict[int, int]) -> bool:
        for i in self.universe:
            if int(vertex.get(i, 0)) not in (0, 1):
                return False

        for i, val in (self.forced or {}).items():
            if int(vertex.get(i, 0)) != int(val):
                return False

        for g in (self.groups_exactly_one or []):
            ones = sum(1 for idx in g if int(vertex.get(idx, 0)) == 1)
            if ones != 1:
                return False

        return True