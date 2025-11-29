# util/eval_official.py (예시 이름)

import numpy as np
from sklearn.metrics import roc_auc_score

def eval_official_on_probs(
    y_ng,          # NG=1, Good=0 (train 전체)
    prob_ng,       # 같은 순서의 NG 확률 (OOF 또는 val)
    max_select=200,
    profit_good=100,
    cost_ng=2000,
):
    """
    공식 대회 수식 그대로:
      - ROC-AUC: Good=1, prob_good = 1 - prob_ng
      - Total Net Profit: prob_ng 낮은 순으로 최대 max_select개 선택
    """
    y_ng = np.asarray(y_ng)
    prob_ng = np.asarray(prob_ng)
    assert len(y_ng) == len(prob_ng)
    n = len(y_ng)

    # Good = 1, NG = 0
    y_good = 1 - y_ng
    prob_good = 1.0 - prob_ng

    # [Task1] ROC-AUC
    roc_auc = roc_auc_score(y_good, prob_good)

    # [Task2] 선택 전략: prob_ng 낮은 순으로 max_select개 선택
    k = min(max_select, n)
    order = np.argsort(prob_ng)   # NG 확률 낮은 순
    decision = np.zeros(n, dtype=bool)
    decision[order[:k]] = True

    is_good = (y_ng == 0)
    is_ng   = (y_ng == 1)

    n_good_sel = np.sum(decision & is_good)
    n_ng_sel   = np.sum(decision & is_ng)

    total_net_profit = profit_good * n_good_sel - cost_ng * n_ng_sel
    # train에서는 k<=max_select로 두고 패널티는 생략

    # 정규화 (공식 그대로)
    part_auc    = max(roc_auc - 0.5, 0.0) / 0.5
    part_profit = max(total_net_profit, 0.0) / 20000.0
    total_score = np.sqrt(part_auc * part_profit)

    return roc_auc, total_net_profit, total_score