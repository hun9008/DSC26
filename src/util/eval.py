# evaluation_utils.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_score_general(
    y_ng,            # NG=1, Good=0
    prob_ng,         # NG일 확률 (predict_proba()[:,1])
    n_select_each=200,
    profit_good=100,
    cost_ng=2000
):
    """
    대회 Task 1 / Task 2 공식과 동일한 평가 함수
    (전체 데이터 기준으로 L/P 반 갈라서 각 n_select_each개 선택)

    Parameters
    ----------
    y_ng : array-like
        라벨, NG=1, Good=0
    prob_ng : array-like
        NG일 확률 (예: clf.predict_proba(X)[:, 1])
    n_select_each : int
        L 구간에서 선택할 개수, P 구간에서 선택할 개수
    profit_good : int
        Good를 선택했을 때의 이익
    cost_ng : int
        NG를 선택했을 때의 비용(손실)

    Returns
    -------
    roc_auc : float
    total_net_profit : float
    total_score : float
    """
    y_ng = np.asarray(y_ng)
    prob_ng = np.asarray(prob_ng)
    n = len(y_ng)
    assert len(prob_ng) == n

    # Good=1, NG=0으로 변환
    y_good = 1 - y_ng
    prob_good = 1.0 - prob_ng

    eval_df = pd.DataFrame({
        "y_good": y_good,
        "prob_ng": prob_ng,
        "prob_good": prob_good
    })

    # L / P 반으로 나누어 의사결정 (각 n_select_each개씩 선택)
    half = n // 2
    eval_df["decision"] = False

    top_L = eval_df.iloc[:half].sort_values("prob_ng").iloc[:n_select_each].index
    top_P = eval_df.iloc[half:].sort_values("prob_ng").iloc[:n_select_each].index

    eval_df.loc[top_L, "decision"] = True
    eval_df.loc[top_P, "decision"] = True

    # Task 1: ROC-AUC (Good=1, prob_good 사용)
    roc_auc = roc_auc_score(eval_df["y_good"], eval_df["prob_good"])

    # Task 2: Total Net Profit
    is_decision = eval_df["decision"]
    is_good = eval_df["y_good"] == 1
    is_ng = eval_df["y_good"] == 0

    total_net_profit = (
        profit_good * (is_decision & is_good).sum()
        - cost_ng * (is_decision & is_ng).sum()
    )

    # Score 정규화
    part_auc = max(roc_auc - 0.5, 0.0) / 0.5
    part_profit = max(total_net_profit, 0.0) / 20000.0

    total_score = np.sqrt(part_auc * part_profit)

    print(f"ROC-AUC Score     : {roc_auc:.6f}")
    print(f"Total Net Profit  : {total_net_profit}")
    print(f"Final Total Score : {total_score:.6f}")

    return roc_auc, total_net_profit, total_score


def calculate_competition_score(
    y_true,
    y_prob,
    k=40,
    profit_good=100,
    cost_ng=2000
):
    """
    266 + 467(test) = 733 (train)
    validation 60개 기준의 간이 평가 함수
    - y_true: NG=1, Good=0
    - NG 확률이 가장 낮은 k개를 decision=True 로 선택

    Returns
    -------
    roc_auc : float
    total_net_profit : float
    total_score : float
    """
    y_true = np.asarray(y_true)    # NG=1, Good=0
    y_ng = y_true
    y_good = 1 - y_ng

    prob_ng = np.asarray(y_prob)
    prob_good = 1.0 - prob_ng

    # Task1: ROC-AUC (Good=1 기준)
    roc_auc = roc_auc_score(y_good, prob_good)

    df_eval = pd.DataFrame({
        "prob_ng": prob_ng,
        "prob_good": prob_good,
        "y_ng": y_ng,
        "y_good": y_good
    })

    # NG 확률 낮은 순으로 k개 선택
    selected = df_eval.nsmallest(k, "prob_ng").copy()
    df_eval["decision"] = False
    df_eval.loc[selected.index, "decision"] = True

    # Task2: Net Profit
    is_decision = df_eval["decision"]
    is_good = df_eval["y_good"] == 1
    is_ng = df_eval["y_good"] == 0

    total_net_profit = (
        profit_good * (is_decision & is_good).sum()
        - cost_ng * (is_decision & is_ng).sum()
    )

    # Score 정규화
    auc_comp = max(roc_auc - 0.5, 0.0) / 0.5
    profit_comp = max(total_net_profit, 0.0) / 20000.0

    total_score = np.sqrt(auc_comp * profit_comp)

    correct_good = (selected["y_ng"] == 0).sum()
    incorrect_ng = (selected["y_ng"] == 1).sum()

    print(f"  ROC-AUC        : {roc_auc:.6f}")
    print(f"  Net Profit     : {total_net_profit}")
    print(f"  Total Score    : {total_score:.6f}")
    print(f"  Selected k     : {k} (Good={correct_good}, NG={incorrect_ng})")

    return roc_auc, total_net_profit, total_score