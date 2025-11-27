import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime

from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

from util.logger import TeeLogger
import sys

# ------------------------------
# 1. 데이터 로딩
# ------------------------------
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
submission_template = pd.read_csv("../data/submission/sample_submission.csv")

train_X = train.drop(columns=['Class']).iloc[:, :-256*3]
train_Y = train['Class'].apply(lambda x: 1 if x == 'NG' else 0)
test_X = test.drop(columns=['ID']).iloc[:, :-256*3]

cat_list = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)))

OE = OneHotEncoder(
    min_frequency=0.01,
    handle_unknown='infrequent_if_exist',
    sparse_output=False
)
OE.fit(train_X[cat_list])

def preprocess(dataset):
    Xc = OE.transform(dataset[cat_list])
    Xn = np.array(dataset[num_list])
    return np.concatenate([Xc, Xn], axis=1)


# ------------------------------
# 2. Logging 시작
# ------------------------------
logger = TeeLogger()
sys.stdout = logger

print("[Main] Start CV-based training")


# ------------------------------
# 3. Stratified K-Fold 적용 (5-fold → approx 587/146 split)
# ------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_roc_list = []
cv_profit_list = []
cv_score_list = []

X_all = preprocess(train_X)
y_all = train_Y.values

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"\n========== Fold {fold_idx + 1} ==========")

    X_train_fold = X_all[train_idx]
    y_train_fold = y_all[train_idx]

    X_val = X_all[val_idx]
    y_val = y_all[val_idx]

    print(f"[Fold {fold_idx+1}] Train size: {X_train_fold.shape[0]}")
    print(f"[Fold {fold_idx+1}] Val size  : {X_val.shape[0]} "
          f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        random_state=42+fold_idx
    )
    model.fit(X_train_fold, y_train_fold)

    # 검증 predict
    val_prob_ng = model.predict_proba(X_val)[:, 1]

    # 평가
    roc, profit, score = calculate_competition_score(
        y_true=y_val,
        y_prob=val_prob_ng,
        # k=15 기본값 그대로 사용 (원하면 k 변경 가능)
    )

    cv_roc_list.append(roc)
    cv_profit_list.append(profit)
    cv_score_list.append(score)


# ------------------------------
# 4. CV 결과 출력
# ------------------------------
print("\n===== Cross Validation Summary =====")
print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")


# ------------------------------
# 5. 전체 train 733개로 최종 모델 학습
# ------------------------------
print("\n[Main] Training final model on full training set")

final_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    random_state=42
)
final_model.fit(X_all, y_all)

# ------------------------------
# 6. Test 467 예측
# ------------------------------
X_test_processed = preprocess(test_X)
pred_test = final_model.predict_proba(X_test_processed)[:, 1]

print(f"[Main] Test prob range: {pred_test.min():.4f} ~ {pred_test.max():.4f}")


# ------------------------------
# 7. 제출 파일 생성 (L/P 각 200개 선택)
# ------------------------------
submission = submission_template.copy()
submission['probability'] = np.concatenate([pred_test, pred_test])
submission['decision'] = False

idx_L = submission.index[:467]
idx_P = submission.index[467:]

# L/P 각각 200개 선택
decision_id_L_list = submission.loc[idx_L].sort_values(
    'probability', ascending=True
).iloc[:200]['ID']

decision_id_P_list = submission.loc[idx_P].sort_values(
    'probability', ascending=True
).iloc[:200]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"../data/submission/my_submission_{timestamp}.csv"

submission.to_csv(save_path, index=False)
print(f"[Main] Saved submission to {save_path}")


# ------------------------------
# 8. 로그 닫기
# ------------------------------
logger.close()
sys.stdout = sys.__stdout__
print(f"[Main] Log saved to: {logger.log_path}")