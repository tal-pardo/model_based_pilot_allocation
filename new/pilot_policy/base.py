from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, Sequence

if TYPE_CHECKING:
    from channel_estimators.lmmse import EstimatorState


class PilotPolicy(Protocol):
    def reset(self) -> None: ...

    def initial_subcarriers(self) -> List[int]:
        """Subcarriers used for bootstrap (length k0)."""
        ...

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        """Choose next unused subcarrier index."""
        ...


def initial_subcarriers_uniform(k0: int, nc: int) -> List[int]:
    """
    Input: k0 initial pilot count, Nc subcarriers.
    Output: evenly spaced initial indices (e.g. k0=2, Nc=32 -> [0, 15]).
    """
    if k0 <= 0:
        raise ValueError("initial_pilot_subcarriers must be positive.")
    if k0 == 1:
        return [0]
    if k0 > nc:
        raise ValueError("initial_pilot_subcarriers cannot exceed n_subcarriers.")
    if k0 == nc:
        return list(range(nc))
    out: List[int] = []
    for i in range(k0):
        if i == 0:
            out.append(0)
        else:
            k = (i * nc) // k0 - 1
            if k <= out[-1]:
                k = out[-1] + 1
            out.append(k)
    return out


def _gap_interior(a: int, b: int, nc: int, *, wrap: bool) -> List[int]:
    if not wrap:
        if b <= a + 1:
            return []
        return list(range(a + 1, b))
    high = list(range(a + 1, nc))
    low = list(range(0, b)) if b > 0 else []
    return high + low


def _pick_in_gap(a: int, b: int, nc: int, interior: Sequence[int], *, wrap: bool) -> int:
    """Input: gap endpoints, interior indices. Output: next subcarrier in that gap."""
    if not interior:
        raise ValueError("empty gap interior")
    if wrap and b == 0 and (nc - 1) in interior and len(interior) >= max(2, nc // 4):
        return nc - 1
    if len(interior) <= 3:
        return max(interior)
    if not wrap:
        return (a + b) // 2
    return interior[len(interior) // 2]


def _collect_gaps(sorted_used: List[int], nc: int) -> List[tuple[int, int, int, bool, List[int]]]:
    """Return list of (gap_len, a, b, wrap, interior)."""
    m = len(sorted_used)
    gaps: List[tuple[int, int, int, bool, List[int]]] = []
    for i in range(m):
        a = sorted_used[i]
        b = sorted_used[(i + 1) % m]
        wrap = i == m - 1 and b <= a
        interior = _gap_interior(a, b, nc, wrap=wrap)
        if not interior:
            continue
        gaps.append((len(interior), a, b, wrap, list(interior)))
    return gaps


def next_uniform_subcarrier(used: Sequence[int], nc: int) -> int:
    """
    Input: list of used subcarrier indices, Nc.
    Output: unused index that best preserves uniform spacing via largest-gap bisection
    (wrap-high edge, local refinement, then upper-half long gaps).
    """
    used_set = set(int(k) for k in used)
    if len(used_set) >= nc:
        raise RuntimeError("All subcarriers already used.")
    if not used_set:
        return 0

    sorted_used = sorted(used_set)
    gaps = _collect_gaps(sorted_used, nc)
    if not gaps:
        unused = [idx for idx in range(nc) if idx not in used_set]
        return unused[0]

    max_len = max(g[0] for g in gaps)
    half = nc // 2

    # Refine tight clusters in the lower half (e.g. fill 14 between 11 and 15).
    small_refine = [
        g for g in gaps if g[0] <= 3 and g[1] >= half - 5 and g[2] <= half - 1
    ]
    if small_refine:
        _, a, b, wrap, interior = max(small_refine, key=lambda g: g[1])
    else:
        long_high = [g for g in gaps if g[0] >= max_len - 1 and g[1] >= half]
        if long_high:
            _, a, b, wrap, interior = max(long_high, key=lambda g: g[1])
        else:
            candidates = [g for g in gaps if g[0] >= max_len - 1]
            _, a, b, wrap, interior = min(candidates, key=lambda g: g[1])

    k = _pick_in_gap(a, b, nc, interior, wrap=wrap)

    if k in used_set:
        unused = [idx for idx in range(nc) if idx not in used_set]
        return unused[0]
    return k
