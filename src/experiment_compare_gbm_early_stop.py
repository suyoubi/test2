"""
对比 HistGradientBoosting 两种设置下的 Walk-forward 回测指标：

  B) early_stopping=True + validation_fraction=0.2 — sklearn 内部 stratified 随机验证切分
  A) early_stopping=False — 当前生产路径（无上述随机切分）

运行顺序: 先 B 后 A，使最终 dual_model_*.pkl 与生产一致。

用法: python3 src/experiment_compare_gbm_early_stop.py
"""

import contextlib
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stock_predictor as sp
from sklearn.ensemble import HistGradientBoostingClassifier


def _install_gbm_factory(early_stop: bool):
    def add_gbm(self, X, y, sample_weight=None, config=None):
        Xf = self._prep(X)
        cfg = config or sp.MODEL_CONFIGS[0]
        if early_stop:
            m = HistGradientBoostingClassifier(
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.20,
                **cfg,
            )
        else:
            m = HistGradientBoostingClassifier(
                max_bins=128, early_stopping=False, **cfg
            )
        m.fit(Xf, y, sample_weight=sample_weight)
        self.models.append(("gbm", m))

    def add_model(self, X, y, sample_weight=None, config=None):
        if self.feature_names is None:
            self.feature_names = list(X.columns)
            self._medians = X.median()
        Xf = X.reindex(columns=self.feature_names).fillna(self._medians)
        for col in Xf.columns:
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce")
        Xf = Xf.fillna(self._medians)
        cfg = config or sp.MODEL_CONFIGS[0]
        if early_stop:
            m = HistGradientBoostingClassifier(
                max_bins=128,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.20,
                **cfg,
            )
        else:
            m = HistGradientBoostingClassifier(
                max_bins=128, early_stopping=False, **cfg
            )
        m.fit(Xf, y, sample_weight=sample_weight)
        self.models.append(m)

    sp.DiverseEnsemble.add_gbm = add_gbm
    sp.EnsemblePredictor.add_model = add_model


def main():
    import run_backtest as rb

    def run_silent():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return rb.run_backtest(skip_fetch=True)

    print("=" * 72)
    print("  GBM early_stopping 对比（同一 Walk-forward 回测）")
    print("=" * 72)

    _install_gbm_factory(True)
    print("\n[1/2] early_stopping=True (内部随机验证切分) ...")
    sys.stdout.flush()
    _, res_b = run_silent()

    _install_gbm_factory(False)
    print("[2/2] early_stopping=False (生产) ...")
    sys.stdout.flush()
    _, res_a = run_silent()

    def row(name, r):
        return (
            f"  {name:30s}  总={r['total_accuracy']:.1%}({r['total_signals']}) "
            f"涨={r['up_accuracy']:.1%}({r['up_count']}) "
            f"跌={r['dn_accuracy']:.1%}({r['dn_count']})  基线={r['baseline']:.1%}"
        )

    print("\n" + "=" * 72)
    print("  结果对比 (A−B 为现配置减旧配置)")
    print("=" * 72)
    print(row("B early_stopping=True", res_b))
    print(row("A early_stopping=False", res_a))
    d_tot = res_a["total_accuracy"] - res_b["total_accuracy"]
    d_up = res_a["up_accuracy"] - res_b["up_accuracy"]
    d_dn = res_a["dn_accuracy"] - res_b["dn_accuracy"]
    print(f"\n  Δ总精度 {d_tot:+.2%}   Δ涨 {d_up:+.2%}   Δ跌 {d_dn:+.2%}")
    print("\n  已用 A 覆盖 results/dual_model_*.pkl。")
    print("=" * 72)


if __name__ == "__main__":
    main()
