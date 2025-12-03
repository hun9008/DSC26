# util/eval_v2.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_score_general(
    y_ng,
    prob_ng,
    official_n_select_each: int = 200,   # 대회에서 L/P 각각 200개
    official_half_size: int = 466,       # 대회에서 한쪽 사이즈 466
    profit_good: int = 100,
    cost_ng: int = 2000,
):
    """
    - y_ng: NG=1, Good=0
    - prob_ng: NG일 확률 (예: clf.predict_proba(X)[:,1])
    - official_n_select_each: 대회 기준 L/P 각각 선택 개수 (기본 200)
    - official_half_size: 대회 기준 L/P 샘플 개수 (기본 466)
    """

    y_ng = np.asarray(y_ng)
    prob_ng = np.asarray(prob_ng)
    n = len(y_ng)
    assert prob_ng.shape[0] == n, "y_ng 과 prob_ng 길이가 다릅니다."

    # Good=1 기준 점수 계산을 위해 변환
    y_good = 1 - y_ng
    prob_good = 1.0 - prob_ng

    eval_df = pd.DataFrame(
        {
            "y_ng": y_ng,
            "y_good": y_good,
            "prob_ng": prob_ng,
            "prob_good": prob_good,
        }
    )

    # -----------------------------
    # 1) L / P 나누기
    # -----------------------------
    half = n // 2
    eval_df["decision"] = False

    if half == 0:
        # 말도 안되게 작은 입력 방어
        roc_auc = roc_auc_score(y_good, prob_good)
        return roc_auc, 0.0, 0.0

    # -----------------------------
    # 2) 공식 비율 기반으로 k_each 계산
    #    ratio = 200 / 466 (기본값)
    # -----------------------------
    official_ratio = official_n_select_each / official_half_size  # ≈ 0.429
    k_each = int(round(official_ratio * half))
    k_each = max(1, min(k_each, half))  # [1, half] 범위 클램프

    # L: 앞 half, P: 뒤 half에서 각각 k_each개씩 선택
    top_L_idx = (
        eval_df.iloc[:half]
        .sort_values("prob_ng", ascending=True)
        .iloc[:k_each]
        .index
    )
    top_P_idx = (
        eval_df.iloc[half:]
        .sort_values("prob_ng", ascending=True)
        .iloc[:k_each]
        .index
    )

    eval_df.loc[top_L_idx, "decision"] = True
    eval_df.loc[top_P_idx, "decision"] = True

    # -----------------------------
    # 3) Task 1: ROC-AUC (Good=1)
    # -----------------------------
    roc_auc = roc_auc_score(eval_df["y_good"], eval_df["prob_good"])

    # -----------------------------
    # 4) Task 2: Net Profit
    # -----------------------------
    is_decision = eval_df["decision"]
    is_good = eval_df["y_good"] == 1
    is_ng = eval_df["y_ng"] == 1

    total_net_profit = (
        profit_good * (is_decision & is_good).sum()
        - cost_ng * (is_decision & is_ng).sum()
    )

    # 선택된 개수 확인용
    selected = eval_df[is_decision]
    selected_good = (selected["y_ng"] == 0).sum()
    selected_ng = (selected["y_ng"] == 1).sum()

    # -----------------------------
    # 5) Score 정규화
    #    - AUC: [0.5,1] → [0,1]
    #    - Profit: [0, max_profit] → [0,1]
    #    - max_profit = 2 * k_each * profit_good
    # -----------------------------
    max_profit = 2 * k_each * profit_good  # L/P 둘 합친 최대 이익

    part_auc = max(roc_auc - 0.5, 0.0) / 0.5
    if max_profit > 0:
        part_profit = max(total_net_profit, 0.0) / max_profit
    else:
        part_profit = 0.0

    total_score = np.sqrt(part_auc * part_profit)

    print(f"ROC-AUC Score     : {roc_auc:.6f}")
    print(f"Total Net Profit  : {total_net_profit}")
    print(f"Final Total Score : {total_score:.6f}")
    print(
        f"Selected total    : {len(selected)} "
        f"(Good={selected_good}, NG={selected_ng}, k_each={k_each})"
    )

    return roc_auc, total_net_profit, total_score


def calculate_competition_score(
    y_true,
    y_prob,
    official_n_select_each: int = 200,
    official_half_size: int = 466,
    profit_good: int = 100,
    cost_ng: int = 2000,
):
    return evaluate_score_general(
        y_ng=y_true,
        prob_ng=y_prob,
        official_n_select_each=official_n_select_each,
        official_half_size=official_half_size,
        profit_good=profit_good,
        cost_ng=cost_ng,
    )