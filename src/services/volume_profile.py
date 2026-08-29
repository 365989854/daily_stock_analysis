# -*- coding: utf-8 -*-
"""
Volume Profile calculator.

Builds an approximate volume-at-price distribution from historical OHLCV bars.

Important:
- This is NOT a holder cost-basis / chip-distribution model.
- It must not be used to fabricate profit ratio or real investor cost basis.
- Designed primarily as a US-stock replacement for A-share chip distribution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _classify_position(
    current_price: Optional[float],
    poc: float,
    vah: float,
    val: float,
) -> str:
    """Describe current price relative to POC / Value Area."""
    if current_price is None or not np.isfinite(current_price):
        return "unknown"

    if current_price > vah:
        return "above_value_area"
    if current_price < val:
        return "below_value_area"
    if current_price >= poc:
        return "inside_value_area_above_poc"
    return "inside_value_area_below_poc"


def _build_high_volume_nodes(
    volumes: np.ndarray,
    edges: np.ndarray,
    total_volume: float,
) -> list[Dict[str, float]]:
    """
    Build contiguous High Volume Nodes.

    Bins at or above the 75th percentile of non-zero bin volume are treated
    as high-volume bins, then adjacent bins are merged into one node.
    """
    positive = volumes[volumes > 0]
    if positive.size == 0 or total_volume <= 0:
        return []

    threshold = float(np.percentile(positive, 75))
    mask = volumes >= threshold

    nodes: list[Dict[str, float]] = []
    start: Optional[int] = None

    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx

        is_last = idx == len(mask) - 1
        if start is not None and ((not active) or is_last):
            end = idx if active and is_last else idx - 1

            node_volume = float(volumes[start : end + 1].sum())
            nodes.append(
                {
                    "low": round(float(edges[start]), 4),
                    "high": round(float(edges[end + 1]), 4),
                    "volume_share": round(node_volume / total_volume, 4),
                }
            )
            start = None

    nodes.sort(key=lambda item: item["volume_share"], reverse=True)
    return nodes[:3]


def calculate_volume_profile(
    df: pd.DataFrame,
    *,
    bins: int = 40,
    value_area_pct: float = 0.70,
    lookback: int = 30,
    current_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calculate an approximate Volume Profile from historical OHLCV bars.

    Each daily bar's volume is distributed across the price bins overlapped
    by that bar's low-high range, proportional to the overlap width.

    Args:
        df:
            DataFrame containing at least high, low and volume columns.
        bins:
            Number of price bins.
        value_area_pct:
            Fraction of total volume included in the Value Area.
        lookback:
            Maximum number of most recent complete bars to use.
        current_price:
            Optional realtime/current price used only for position
            classification. It does not affect the profile calculation.

    Returns:
        Dict containing POC, VAH, VAL and HVNs, or None if data is insufficient.
    """
    if df is None or df.empty:
        return None

    if bins < 10:
        bins = 10
    elif bins > 200:
        bins = 200

    if lookback < 5:
        lookback = 5

    value_area_pct = max(0.50, min(float(value_area_pct), 0.95))

    required = {"high", "low", "volume"}
    if not required.issubset(df.columns):
        return None

    data = df.copy()

    for column in ("high", "low", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "close" in data.columns:
        data["close"] = pd.to_numeric(data["close"], errors="coerce")

    data = data.dropna(subset=["high", "low", "volume"])

    data = data[
        (data["high"] >= data["low"])
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["volume"] > 0)
    ]

    if data.empty:
        return None

    data = data.tail(lookback).copy()

    price_min = float(data["low"].min())
    price_max = float(data["high"].max())

    if not np.isfinite(price_min) or not np.isfinite(price_max):
        return None

    if price_max <= price_min:
        return None

    edges = np.linspace(price_min, price_max, bins + 1)
    profile = np.zeros(bins, dtype=float)

    for row in data.itertuples(index=False):
        low = float(getattr(row, "low"))
        high = float(getattr(row, "high"))
        volume = float(getattr(row, "volume"))

        if not (
            np.isfinite(low)
            and np.isfinite(high)
            and np.isfinite(volume)
            and volume > 0
        ):
            continue

        bar_range = high - low

        # Flat bar: put the whole volume into the containing price bin.
        if bar_range <= 0:
            idx = int(np.searchsorted(edges, low, side="right") - 1)
            idx = max(0, min(idx, bins - 1))
            profile[idx] += volume
            continue

        first_bin = int(np.searchsorted(edges, low, side="right") - 1)
        last_bin = int(np.searchsorted(edges, high, side="left"))

        first_bin = max(0, min(first_bin, bins - 1))
        last_bin = max(0, min(last_bin, bins - 1))

        overlaps: list[tuple[int, float]] = []
        overlap_total = 0.0

        for idx in range(first_bin, last_bin + 1):
            overlap_low = max(low, float(edges[idx]))
            overlap_high = min(high, float(edges[idx + 1]))
            overlap = max(0.0, overlap_high - overlap_low)

            if overlap > 0:
                overlaps.append((idx, overlap))
                overlap_total += overlap

        if overlap_total <= 0:
            midpoint = (low + high) / 2.0
            idx = int(np.searchsorted(edges, midpoint, side="right") - 1)
            idx = max(0, min(idx, bins - 1))
            profile[idx] += volume
            continue

        for idx, overlap in overlaps:
            profile[idx] += volume * (overlap / overlap_total)

    total_volume = float(profile.sum())

    if total_volume <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) / 2.0

    # POC = price bin with the largest accumulated volume.
    poc_idx = int(np.argmax(profile))
    poc = float(centers[poc_idx])

    # Build a contiguous Value Area around POC.
    target_volume = total_volume * value_area_pct
    accumulated = float(profile[poc_idx])

    left = poc_idx - 1
    right = poc_idx + 1
    selected_low = poc_idx
    selected_high = poc_idx

    while accumulated < target_volume and (left >= 0 or right < bins):
        left_volume = float(profile[left]) if left >= 0 else -1.0
        right_volume = float(profile[right]) if right < bins else -1.0

        if right_volume > left_volume:
            accumulated += max(right_volume, 0.0)
            selected_high = right
            right += 1
        else:
            accumulated += max(left_volume, 0.0)
            selected_low = left
            left -= 1

    val = float(edges[selected_low])
    vah = float(edges[selected_high + 1])

    # If no realtime current price was supplied, use the latest close.
    if current_price is None and "close" in data.columns:
        valid_close = data["close"].dropna()
        if not valid_close.empty:
            latest_close = float(valid_close.iloc[-1])
            if np.isfinite(latest_close) and latest_close > 0:
                current_price = latest_close

    normalized_current_price: Optional[float] = None
    if current_price is not None:
        try:
            numeric_price = float(current_price)
            if np.isfinite(numeric_price) and numeric_price > 0:
                normalized_current_price = numeric_price
        except (TypeError, ValueError):
            pass

    high_volume_nodes = _build_high_volume_nodes(
        profile,
        edges,
        total_volume,
    )

    return {
        "lookback": int(len(data)),
        "bins": int(bins),
        "value_area_pct": round(value_area_pct, 4),
        "poc": round(poc, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "current_price": (
            round(normalized_current_price, 4)
            if normalized_current_price is not None
            else None
        ),
        "position": _classify_position(
            normalized_current_price,
            poc,
            vah,
            val,
        ),
        "high_volume_nodes": high_volume_nodes,
        "total_volume": round(total_volume, 2),
        "source": "derived_ohlcv",
    }
