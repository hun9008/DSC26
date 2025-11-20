import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def evaluate_score_general(
    y_good,          # Good=1, NG=0
    prob_good,       # Good일 확률 (predict_proba()[:,1])
    n_select_each=200,
    profit_good=100,
    cost_ng=2000,
):
    """
    대회 공식 스코어 형태를 일반적으로 계산하는 함수.
    - y_good: Good=1, NG=0
    - prob_good: Good 확률
    - L/P 구간: 앞 절반 / 뒤 절반 기준
    - 각 구간에서 prob_good가 큰 순으로 n_select_each개 선정
    """
    y_good = np.asarray(y_good)
    prob_good = np.asarray(prob_good)
    n = len(y_good)
    assert len(prob_good) == n

    prob_ng = 1.0 - prob_good

    eval_df = pd.DataFrame({
        "y_good": y_good,
        "prob_good": prob_good,
        "prob_ng": prob_ng,
    })

    half = n // 2
    eval_df["decision"] = False

    # 각 구간에서 Good 확률이 큰 순으로 선택
    top_L = eval_df.iloc[:half].sort_values("prob_good", ascending=False).iloc[:n_select_each].index
    top_P = eval_df.iloc[half:].sort_values("prob_good", ascending=False).iloc[:n_select_each].index

    eval_df.loc[top_L, "decision"] = True
    eval_df.loc[top_P, "decision"] = True

    # ROC-AUC (Good=1, prob_good 사용)
    roc_auc = roc_auc_score(eval_df["y_good"], eval_df["prob_good"])

    is_decision = eval_df["decision"]
    is_good = eval_df["y_good"] == 1
    is_ng = eval_df["y_good"] == 0

    total_net_profit = (
        profit_good * (is_decision & is_good).sum()
        - cost_ng * (is_decision & is_ng).sum()
    )

    # 이론적 최대 이익 = 모두 Good이라고 가정했을 때
    n_decision = int(is_decision.sum())
    max_profit = profit_good * n_decision if n_decision > 0 else profit_good

    part_auc = max(roc_auc - 0.5, 0) / 0.5          # 0.5~1 -> 0~1
    part_profit = max(total_net_profit, 0) / max_profit if max_profit > 0 else 0.0

    total_score = np.sqrt(part_auc * part_profit)   # 둘 다 [0,1] 이므로 [0,1]

    print(f"ROC-AUC Score        : {roc_auc:.6f}")
    print(f"Total Net Profit     : {total_net_profit}")
    print(f"Final Total Score    : {total_score:.6f}")

    return roc_auc, total_net_profit, total_score


# ==================== 데이터 로딩 및 전처리 ====================

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/sample_submission.csv")

# 이미지 좌표(p,x,y) 뒤의 256*3은 제거
train_X = train.drop(columns=["Class"]).iloc[:, :-256 * 3]
test_X = test.drop(columns=["ID"]).iloc[:, :-256 * 3]

# Good=1, NG=0
train_Y_good = (train["Class"] == "Good").astype(int)

cat_list = train_X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)))

OE = OneHotEncoder(
    min_frequency=0.01,
    handle_unknown="infrequent_if_exist",
    sparse_output=False,
)

if len(cat_list) > 0:
    OE.fit(train_X[cat_list])
else:
    # 범주형이 없는 경우 더미 fit
    OE.fit(pd.DataFrame(index=train_X.index))


def preprocess(dataset: pd.DataFrame) -> np.ndarray:
    if len(cat_list) > 0:
        Xc = OE.transform(dataset[cat_list])
    else:
        Xc = np.zeros((len(dataset), 0), dtype=float)
    Xn = np.array(dataset[num_list], dtype=float)
    return np.concatenate([Xc, Xn], axis=1)


X_train = preprocess(train_X)
X_test = preprocess(test_X)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("Good / NG count:", train_Y_good.sum(), "/", len(train_Y_good) - train_Y_good.sum())

# ==================== 모델 정의 ====================

models = {}

models["DecisionTree"] = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
)

models["RandomForest"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    n_jobs=-1,
    random_state=42,
)

models["ExtraTrees"] = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

models["GradientBoosting"] = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

models["AdaBoost"] = AdaBoostClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
)

models["HistGB"] = HistGradientBoostingClassifier(
    max_depth=10,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
)

models["XGBoost"] = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

models["LightGBM"] = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)

models["CatBoost"] = CatBoostClassifier(
    iterations=400,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=False,
    random_state=42,
)

# ==================== 학습 & 모델별 성능 ====================

train_probs = {}
test_probs = {}

for name, clf in models.items():
    print("=" * 60)
    print(f"Model: {name}")
    clf.fit(X_train, train_Y_good)

    # Good(1)일 확률
    train_prob_good = clf.predict_proba(X_train)[:, 1]
    test_prob_good = clf.predict_proba(X_test)[:, 1]

    train_probs[name] = train_prob_good
    test_probs[name] = test_prob_good

    print("Train performance:")
    evaluate_score_general(
        y_good=train_Y_good,
        prob_good=train_prob_good,
        n_select_each=200,
        profit_good=100,
        cost_ng=2000,
    )

# ==================== 앙상블(평균) ====================

print("=" * 60)
print("Ensemble (average of all models)")

train_prob_ensemble = np.mean(
    np.column_stack([train_probs[name] for name in models.keys()]),
    axis=1,
)
test_prob_ensemble = np.mean(
    np.column_stack([test_probs[name] for name in models.keys()]),
    axis=1,
)

print("Train performance (Ensemble):")
roc_auc_ens, total_net_profit_ens, total_score_ens = evaluate_score_general(
    y_good=train_Y_good,
    prob_good=train_prob_ensemble,
    n_select_each=200,
    profit_good=100,
    cost_ng=2000,
)

# ==================== 제출 파일 생성 ====================

# probability = Good 확률을 그대로 사용
submission["probability"] = np.concatenate([test_prob_ensemble, test_prob_ensemble])
submission["decision"] = False

n_sub = len(submission)
half_sub = n_sub // 2

idx_L_sub = submission.index[:half_sub]
idx_P_sub = submission.index[half_sub:]

# 각 구간에서 Good 확률이 큰 순으로 200개씩 선택
decision_id_L_list = (
    submission.loc[idx_L_sub]
    .sort_values("probability", ascending=False)
    .iloc[:200]["ID"]
)
decision_id_P_list = (
    submission.loc[idx_P_sub]
    .sort_values("probability", ascending=False)
    .iloc[:200]["ID"]
)

submission.loc[submission["ID"].isin(decision_id_L_list), "decision"] = True
submission.loc[submission["ID"].isin(decision_id_P_list), "decision"] = True

submission.to_csv("./data/ensemble_submission.csv", index=False)

print("=" * 60)
print("Final Ensemble Train Metrics")
print(f"ROC-AUC Score        : {roc_auc_ens:.6f}")
print(f"Total Net Profit     : {total_net_profit_ens}")
print(f"Final Total Score    : {total_score_ens:.6f}")
print("Saved submission to ./data/ensemble_submission.csv")

# ==================== 제출 파일 디버깅 출력 ====================

print("\n[DEBUG] Submission head:")
print(submission.head())

print("\n[DEBUG] Submission tail:")
print(submission.tail())

print("\n[DEBUG] probability describe:")
print(submission["probability"].describe())

print("\n[DEBUG] decision value_counts:")
print(submission["decision"].value_counts())

print("\n[DEBUG] L / P 구간별 decision 개수:")
print("L 구간 decision=True 개수:",
      submission.iloc[:half_sub]["decision"].sum())
print("P 구간 decision=True 개수:",
      submission.iloc[half_sub:]["decision"].sum())

print("\n[DEBUG] Null 체크:")
print(submission.isna().sum())