import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score


def evaluate_score_general(
    y_ng,
    prob_ng,
    n_select_each=200,
    profit_good=100,
    cost_ng=150
):
    """
    대회에서 주어진 평가 스코어 계산 함수
    (L/P 반으로 쪼개서 의사결정, ROC-AUC + Profit 기반 Score)
    """
    y_ng = np.asarray(y_ng)
    prob_ng = np.asarray(prob_ng)
    n = len(y_ng)
    assert len(prob_ng) == n

    # Good=1, NG=0
    y_good = 1 - y_ng
    prob_good = 1.0 - prob_ng

    eval_df = pd.DataFrame({
        "y_good": y_good,
        "prob_ng": prob_ng,
        "prob_good": prob_good
    })

    # L / P 반으로 나누어 의사결정
    half = n // 2
    eval_df["decision"] = False

    top_L = eval_df.iloc[:half].sort_values("prob_ng").iloc[:n_select_each].index
    top_P = eval_df.iloc[half:].sort_values("prob_ng").iloc[:n_select_each].index

    eval_df.loc[top_L, "decision"] = True
    eval_df.loc[top_P, "decision"] = True

    # ROC-AUC (Good=1, prob_good 사용)
    roc_auc = roc_auc_score(eval_df["y_good"], eval_df["prob_good"])

    # 이익 계산
    is_decision = eval_df["decision"]
    is_good = eval_df["y_good"] == 1
    is_ng = eval_df["y_good"] == 0

    total_net_profit = (
        profit_good * (is_decision & is_good).sum()
        - cost_ng * (is_decision & is_ng).sum()
    )

    n_decision = int(is_decision.sum())
    max_profit = profit_good * n_decision if n_decision > 0 else profit_good

    part_auc = max(roc_auc - 0.5, 0) / 0.5
    part_profit = max(total_net_profit, 0) / max_profit if max_profit > 0 else 0.0

    total_score = np.sqrt(part_auc * part_profit)

    print(f"ROC-AUC Score        : {roc_auc:.6f}")
    print(f"Total Net Profit     : {total_net_profit}")
    print(f"Final Total Score    : {total_score:.6f}")

    return roc_auc, total_net_profit, total_score


def make_fixed_validation_indices(
    y_series,
    n_ng_val=15,
    n_good_val=45,
    seed=42,
):
    """
    y_series: pandas Series, 값은 0(Good), 1(NG)
    NG 15, Good 45 로 총 60개 validation index 생성
    나머지는 train index 로 반환
    """
    rng = np.random.RandomState(seed)
    y = y_series.values
    all_idx = np.arange(len(y))

    ng_idx = all_idx[y == 1]
    good_idx = all_idx[y == 0]

    if len(ng_idx) < n_ng_val or len(good_idx) < n_good_val:
        raise ValueError("Validation 에 필요한 NG 또는 Good 샘플 수가 부족합니다.")

    rng.shuffle(ng_idx)
    rng.shuffle(good_idx)

    val_ng_idx = ng_idx[:n_ng_val]
    val_good_idx = good_idx[:n_good_val]

    val_idx = np.concatenate([val_ng_idx, val_good_idx])
    rng.shuffle(val_idx)

    train_idx = np.setdiff1d(all_idx, val_idx)

    return train_idx, val_idx


# ---------------------- 메인 파이프라인 ---------------------- #

# 1. 데이터 로드
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/sample_submission.csv")

# 2. 기본 피처 분리 (좌표/압력 256*3 제거)
train_X = train.drop(columns=['Class']).iloc[:, :-256*3]
train_Y_ng = (train['Class'] == 'NG').astype(int)   # NG=1, Good=0
test_X = test.drop(columns=['ID']).iloc[:, :-256*3]

# 3. 범주형 / 수치형 컬럼 구분
cat_list = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)))

# 4. OneHotEncoder 학습 (train 전체 기준)
OE = OneHotEncoder(
    min_frequency=0.01,
    handle_unknown='infrequent_if_exist',
    sparse_output=False
)
if len(cat_list) > 0:
    OE.fit(train_X[cat_list])
else:
    OE.fit(pd.DataFrame(index=train_X.index))


def preprocess(dataset):
    """train/test 에 공통으로 사용하는 전처리 함수"""
    if len(cat_list) > 0:
        Xc = OE.transform(dataset[cat_list])
    else:
        Xc = np.zeros((len(dataset), 0))
    Xn = np.array(dataset[num_list])
    return np.concatenate([Xc, Xn], axis=1)


# 5. 전체 encodings 미리 계산
X_all = preprocess(train_X)
X_test = preprocess(test_X)

print(f"Encoded train shape: {X_all.shape}")
print(f"Encoded test  shape: {X_test.shape}")

# 6. Cross Validation 설정
n_splits = 5             # 몇 번 반복할지 (원하면 10 등으로 변경 가능)
n_ng_val = 15
n_good_val = 45

cv_roc_list = []
cv_profit_list = []
cv_score_list = []

print("\n===== Cross Validation (each val: 60 samples, NG:15, Good:45) =====")
for fold in range(n_splits):
    seed = 42 + fold   # fold마다 다른 seed
    train_idx, val_idx = make_fixed_validation_indices(
        train_Y_ng,
        n_ng_val=n_ng_val,
        n_good_val=n_good_val,
        seed=seed
    )

    X_train_fold = X_all[train_idx]
    y_train_fold = train_Y_ng.values[train_idx]

    X_val_fold = X_all[val_idx]
    y_val_fold = train_Y_ng.values[val_idx]

    print(f"\n[Fold {fold+1}/{n_splits}] "
          f"train_size={X_train_fold.shape[0]}, val_size={X_val_fold.shape[0]} "
          f"(NG={ (y_val_fold==1).sum() }, Good={ (y_val_fold==0).sum() })")

    # 7. fold별 모델 학습
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        random_state=seed
    )
    model.fit(X_train_fold, y_train_fold)

    # 8. fold별 validation 평가
    val_prob_ng = model.predict_proba(X_val_fold)[:, 1]
    roc_auc, total_net_profit, total_score = evaluate_score_general(
        y_ng=y_val_fold,
        prob_ng=val_prob_ng,
        n_select_each=200,      # 대회 스펙 유지 (샘플이 60이라 실제로는 half 전체 선택되는 형태)
        profit_good=100,
        cost_ng=150
    )

    cv_roc_list.append(roc_auc)
    cv_profit_list.append(total_net_profit)
    cv_score_list.append(total_score)

# 9. CV 평균 결과 출력
print("\n===== CV Summary ({} folds) =====".format(n_splits))
print(f"ROC-AUC  mean: {np.mean(cv_roc_list):.6f}  std: {np.std(cv_roc_list):.6f}")
print(f"Profit   mean: {np.mean(cv_profit_list):.2f}  std: {np.std(cv_profit_list):.2f}")
print(f"Score    mean: {np.mean(cv_score_list):.6f}  std: {np.std(cv_score_list):.6f}")

# 10. 최종 모델: 전체 train 으로 재학습
final_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    random_state=42
)
final_model.fit(X_all, train_Y_ng)

# (옵션) train 전체 기준 성능도 보고 싶으면 아래 주석 해제
# train_prob_ng_full = final_model.predict_proba(X_all)[:, 1]
# print("\n[Full train evaluation]")
# evaluate_score_general(
#     y_ng=train_Y_ng,
#     prob_ng=train_prob_ng_full,
#     n_select_each=200,
#     profit_good=100,
#     cost_ng=2000
# )

# 11. Test 예측
pred_prob_ng = final_model.predict_proba(X_test)[:, 1]

# 12. 제출 파일 생성
submission['probability'] = np.concatenate([pred_prob_ng, pred_prob_ng])
submission['decision'] = False

n_sub = len(submission)
half_sub = n_sub // 2

idx_L_sub = submission.index[:half_sub]
idx_P_sub = submission.index[half_sub:]

decision_id_L_list = submission.loc[idx_L_sub].sort_values(
    'probability', ascending=True
).iloc[:170]['ID']
decision_id_P_list = submission.loc[idx_P_sub].sort_values(
    'probability', ascending=True
).iloc[:170]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

submission.to_csv("./data/my_submission_cv.csv", index=False)
print("\nSaved submission to ./data/my_submission_cv.csv")