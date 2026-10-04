"""Statistical-arbitrage pairs trading (built-in plugin).

``fit`` picks the pairs: either explicit ``options.pairs`` or a cointegration
screen on the fit window. ``positions_batch`` is fully vectorized over pairs ×
(entry, exit, stop) combos for one (beta_window, z_window) group.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.data.store import MarketData
from quantlab.strategies.pairs.engine import Thresholds, hedge_and_zscore, target_positions
from quantlab.strategies.pairs.screening import engle_granger, screen_pairs
from quantlab.strategy.base import Params, Strategy


class PairsStrategy(Strategy):
    name = "pairs"
    group_by = ("beta_window", "z_window")
    default_options: dict[str, Any] = {
        "pairs": None,  # [[Y, X], ...]; screened when omitted
        "screen": {"min_corr": 0.7, "max_pvalue": 0.05, "max_half_life": 60.0, "max_pairs": 20},
        "coint_pvalue": 0.10,  # validation: per-window Engle-Granger threshold
        "min_coint_stability": 0.5,  # validation: share of windows a pair must stay cointegrated
        "min_stable_pairs": 2,  # validation gate
    }

    def fit(self, data: MarketData) -> dict[str, Any]:
        if self.options.get("pairs"):
            pairs, screened = [list(p) for p in self.options["pairs"]], []
        else:
            cands = screen_pairs(data.close, **{**self.default_options["screen"], **self.options.get("screen", {})})
            if not cands:
                raise RuntimeError("pair screen found no cointegrated pairs; relax strategy.options.screen")
            pairs, screened = [[c.y, c.x] for c in cands], [c.to_dict() for c in cands]
        return {"pairs": pairs, "symbols": sorted({s for p in pairs for s in p}), "screened": screened}

    def valid(self, p: Params) -> bool:
        return p["exit_z"] < p["entry_z"] < p["stop_z"]

    def warmup(self, p: Params) -> int:
        return int(p["beta_window"] + p["z_window"])

    def _legs(self, data: MarketData) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
        pairs = [tuple(p) for p in self.state["pairs"]]
        col = {s: i for i, s in enumerate(data.symbols)}
        return pairs, np.array([col[y] for y, _ in pairs]), np.array([col[x] for _, x in pairs])

    def signals(self, data: MarketData, beta_window: int, z_window: int) -> tuple[np.ndarray, np.ndarray]:
        pairs, iy, ix = self._legs(data)
        logp = np.log(data.close)
        return hedge_and_zscore(logp.iloc[:, iy], logp.iloc[:, ix], beta_window, z_window)

    def positions_batch(self, data: MarketData, params_list: list[Params]) -> np.ndarray:
        groups = {(p["beta_window"], p["z_window"]) for p in params_list}
        if len(groups) > 1:  # called outside the planner's grouping: split and recombine
            out = np.zeros((len(data), len(data.symbols), len(params_list)))
            for g in groups:
                idx = [i for i, p in enumerate(params_list) if (p["beta_window"], p["z_window"]) == g]
                out[:, :, idx] = self.positions_batch(data, [params_list[i] for i in idx])
            return out
        (bw, zw), = groups
        pairs, iy, ix = self._legs(data)
        beta, z = self.signals(data, bw, zw)
        th = Thresholds(*(np.array([p[k] for p in params_list], dtype=float) for k in ("entry_z", "exit_z", "stop_z")))
        pos = target_positions(z, th).astype(np.float64)  # (T, P, C)
        b = np.nan_to_num(beta)[:, :, None]
        wy = pos / (1.0 + np.abs(b)) / len(pairs)
        wx = -pos * b / (1.0 + np.abs(b)) / len(pairs)
        W = np.zeros((len(data), len(data.symbols), len(params_list)))
        for i in range(len(pairs)):
            W[:, iy[i], :] += wy[:, i, :]
            W[:, ix[i], :] += wx[:, i, :]
        return W

    def positions(self, data: MarketData, params: Params) -> np.ndarray:
        return self.positions_batch(data, [params])[:, :, 0]

    def stability(self, data: MarketData, windows: list[tuple[int, int]]) -> dict[str, float]:
        logp = np.log(data.close)
        out = {}
        for y, x in self.state["pairs"]:
            ok = [engle_granger(logp[y].to_numpy()[a:b], logp[x].to_numpy()[a:b])[0] < self.options["coint_pvalue"]
                  for a, b in windows]
            out[f"{y}/{x}"] = float(np.mean(ok))
        return out

    def checks(self, data, params, windows):
        stab = self.stability(data, windows)
        n = sum(v >= self.options["min_coint_stability"] for v in stab.values())
        need = self.options["min_stable_pairs"]
        return [{"check": "n_stable_pairs", "value": n, "rule": f">= {need}", "passed": n >= need,
                 "detail": stab}]

    def describe(self, data, params):
        beta, z = self.signals(data, params["beta_window"], params["z_window"])
        pos = self.positions_batch(data, [params])
        pairs, iy, _ = self._legs(data)
        return {"pairs": [
            {"y": y, "x": x, "latest_beta": round(float(beta[-1, i]), 4), "latest_z": round(float(z[-1, i]), 3),
             "position": int(np.sign(pos[-1, iy[i], 0]))}
            for i, (y, x) in enumerate(pairs)
        ]}
