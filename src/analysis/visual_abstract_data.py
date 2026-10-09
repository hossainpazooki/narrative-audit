"""Generate the two mini-series shown in the visual abstract from the real
committed data, emphasizing the recent change:
  Path 1: NeurIPS Flesch Reading Ease (readability_flesch_ease.csv, neurips).
  Path 2: LLM-as-judge z-score, averaged over the three prompts.

Pre-2022 uses the smoothed trend (centered rolling mean), sampled every 4
years. 2022 onward uses the raw yearly values, sampled every 2 years, so the
sparse recent points show the sharp recent movement without smoothing it out.

Outputs:
  data/pgfplots/va_flesch.csv   year, flesch
  data/pgfplots/va_zscore.csv   year, z

Run from repo root:  python src/analysis/visual_abstract_data.py
"""
from __future__ import annotations

import os

import pandas as pd

_D = os.environ.get("AA_DATA", "data")
PG = os.path.join(_D, "pgfplots")
os.makedirs(PG, exist_ok=True)
WIN = 3  # centered rolling-mean window for the pre-2022 trend
PRE_YEARS = [1987, 1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019]  # every 4y, smoothed


def series_points(yearly: pd.Series, pre_years: list[int],
                  post_years: list[int]) -> list[tuple[int, float]]:
    """Smoothed value at pre_years (< 2022), then raw value at post_years."""
    smoothed = yearly.rolling(WIN, center=True, min_periods=1).mean()
    pts = [(y, round(float(smoothed.loc[y]), 3)) for y in pre_years if y in smoothed.index]
    pts += [(y, round(float(yearly.loc[y]), 3)) for y in post_years if y in yearly.index]
    return pts


def main() -> None:
    # --- Path 1: Flesch ---
    fl = pd.read_csv(f"{PG}/readability_flesch_ease.csv", usecols=["year", "neurips"]).dropna()
    fl_yearly = fl.set_index("year")["neurips"].sort_index()
    fl_pts = series_points(fl_yearly, PRE_YEARS, post_years=[2022, 2024, 2025])
    pd.DataFrame(fl_pts, columns=["year", "flesch"]).to_csv(f"{PG}/va_flesch.csv", index=False)

    # --- Path 2: z-score averaged over the three prompts ---
    prompts = ["simple", "ascb", "own_reasoning"]
    z = None
    for p in prompts:
        d = pd.read_csv(f"{PG}/llm_scores_model_avg_{p}.csv", usecols=["year", "avg_z"]).rename(
            columns={"avg_z": p})
        z = d if z is None else z.merge(d, on="year")
    z_yearly = z.set_index("year")[prompts].mean(axis=1).sort_index()
    z_pts = series_points(z_yearly, PRE_YEARS, post_years=[2022, 2023, 2024])  # not scored in 2025
    pd.DataFrame(z_pts, columns=["year", "z"]).to_csv(f"{PG}/va_zscore.csv", index=False)

    print("va_flesch.csv:")
    print(pd.DataFrame(fl_pts, columns=["year", "flesch"]).to_string(index=False))
    print("\nva_zscore.csv:")
    print(pd.DataFrame(z_pts, columns=["year", "z"]).to_string(index=False))


if __name__ == "__main__":
    main()
