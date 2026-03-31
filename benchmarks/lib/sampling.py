"""Sampling strategies for benchmark configurations."""

from __future__ import annotations


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _evenly_spaced(values: list, count: int) -> list:
    """Select evenly spaced elements from a list."""
    if count <= 0:
        return []
    if count == 1:
        return [values[0]]
    n = len(values)
    idxs = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen: set[int] = set()
    sampled: list = []
    for idx in idxs:
        if idx not in seen:
            sampled.append(values[idx])
            seen.add(idx)
    if len(sampled) < count:
        for idx in range(n):
            if idx not in seen:
                sampled.append(values[idx])
                seen.add(idx)
                if len(sampled) == count:
                    break
    return sampled


def _choose_grid_counts(t_count: int, kc_count: int, max_points: int) -> tuple[int, int]:
    """Choose balanced T and KC sample counts for a given max_points budget."""
    best_t, best_kc = 1, 1
    best_prod = 1
    best_balance = float("inf")
    for t_idx in range(1, t_count + 1):
        kc_idx = min(kc_count, max_points // t_idx)
        if kc_idx < 1:
            continue
        prod = t_idx * kc_idx
        balance = abs((t_idx / t_count) - (kc_idx / kc_count))
        if prod > best_prod or (prod == best_prod and balance < best_balance):
            best_t, best_kc = t_idx, kc_idx
            best_prod = prod
            best_balance = balance
    return best_t, best_kc


def sample_configurations(
    T_list: list[int],
    K_list: list[int],
    C_list: list[int],
    B: int,
    max_points: int,
) -> list[tuple[int, int, int]]:
    """Sample (T, K, C) configs by T and K*C, anchored at min/max BTKC."""
    full_configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]
    if max_points <= 0 or max_points >= len(full_configs):
        return full_configs

    t_values = sorted(set(T_list))
    kc_to_pairs: dict[int, list[tuple[int, int]]] = {}
    for K in K_list:
        for C in C_list:
            kc_to_pairs.setdefault(K * C, []).append((K, C))
    kc_values = sorted(kc_to_pairs.keys())

    t_count = len(t_values)
    kc_count = len(kc_values)
    t_sample_count, kc_sample_count = _choose_grid_counts(t_count, kc_count, max_points)

    t_samples = _evenly_spaced(t_values, t_sample_count)
    kc_samples = _evenly_spaced(kc_values, kc_sample_count)

    sampled_pairs: set[tuple[int, int]] = {(T, KC) for T in t_samples for KC in kc_samples}
    sampled_pairs.add((t_values[0], kc_values[0]))
    sampled_pairs.add((t_values[-1], kc_values[-1]))

    if len(sampled_pairs) < max_points:
        all_pairs = [(T, KC) for T in t_values for KC in kc_values]
        all_pairs.sort(key=lambda pair: B * pair[0] * pair[1])
        for pair in all_pairs:
            if len(sampled_pairs) >= max_points:
                break
            sampled_pairs.add(pair)

    t_index_map = {T: idx for idx, T in enumerate(t_values)}
    kc_index_map = {KC: idx for idx, KC in enumerate(kc_values)}

    configs: list[tuple[int, int, int]] = []
    for T, KC in sorted(sampled_pairs, key=lambda pair: (pair[0], pair[1])):
        pairs = kc_to_pairs[KC]
        pair_idx = (t_index_map[T] + kc_index_map[KC]) % len(pairs)
        K, C = pairs[pair_idx]
        configs.append((T, K, C))

    return configs
