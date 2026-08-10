"""Map offline-evaluation metrics to the paper's result tables."""

from __future__ import annotations

RQ1_COLUMNS: dict[str, str] = {
    "AIME-25": "aime2025_pass1",
    "AIME-26": "aime2026_pass1",
    "GPQA-Diamond": "gpqa_diamond_avg_at_n",
    "LiveCodeBench": "livecodebench_pass_at_1",
    # Golden Goose rollups over the 100 individual rg:*-hard tasks, emitted by
    # GemEvalResult.to_metrics_dict as gem_eval/category_rg_<bucket>/win_rate.
    "RG:Math": "gem_eval/category_rg_math/win_rate",
    "RG:Algorithmic": "gem_eval/category_rg_algorithmic/win_rate",
    "RG:Cognition": "gem_eval/category_rg_cognition/win_rate",
    "RG:Logic": "gem_eval/category_rg_logic/win_rate",
}

RQ2_COLUMNS: dict[str, str] = {
    "BFCL-v4": "bfcl_overall_acc",
    "Tau2-Retail": "gem_eval/tau2_retail_base/pass_at_1",
    "Tau2-Airline": "gem_eval/tau2_airline_base/pass_at_1",
    "Tau2-Telecom": "gem_eval/tau2_telecom_base/pass_at_1",
    "ACEBench-N": "acebench/en/normal",
    "ACEBench-S": "acebench/en/special",
    "ACEBench-A": "acebench/en/agent",
    "ACEBench": "acebench/en/overall",
}

COLUMN_MAPS = {
    "RQ1": RQ1_COLUMNS,
    "RQ2": RQ2_COLUMNS,
}
