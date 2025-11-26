import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_auc_score
from datetime import datetime

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
submission = pd.read_csv("../data/submission/sample_submission.csv")

# display(train.head())
# display(test.head())
# display(submission.head())

train_X = train.drop(columns=['Class']).iloc[:,:-256*3]
train_Y = train['Class'].apply(lambda x:1 if x=='NG' else 0)
test_X = test.drop(columns=['ID']).iloc[:,:-256*3]

# display(train_X.head())
# display(train_Y.head())
# display(test_X.head())

cat_list = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)))

OE = OneHotEncoder(min_frequency=0.01,handle_unknown='infrequent_if_exist',sparse_output=False)
OE.fit(train_X[cat_list])

def preprocess(dataset):
    Xc = OE.transform(dataset[cat_list])
    Xn = np.array(dataset[num_list])
    return np.concatenate([Xc, Xn], axis=1)

n_estimators=200
random_state=42
n_jobs=-1

model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
model.fit(preprocess(train_X), train_Y)

pred = model.predict_proba(preprocess(test_X))[:,1]

train_pred = model.predict_proba(preprocess(train_X))[:, 1]

eval_df = pd.DataFrame({
    "y": train_Y.values,   # 정답 레이블 (0/1)
    "prob": train_pred
})

n = len(eval_df)
half = n // 2

eval_df["decision"] = False

# L 구간
top_L = eval_df.iloc[:half].sort_values("prob").iloc[:200].index
# P 구간
top_P = eval_df.iloc[half:].sort_values("prob").iloc[:200].index

eval_df.loc[top_L, "decision"] = True
eval_df.loc[top_P, "decision"] = True

roc_auc = roc_auc_score(eval_df["y"], eval_df["prob"])

# ------------------------
# 5) Total Net Profit 계산
#    y=1 => Good, y=0 => NG 라고 가정
# ------------------------
is_decision = eval_df["decision"]
is_good = eval_df["y"] == 1
is_ng = eval_df["y"] == 0

total_net_profit = (
    100 * (is_decision & is_good).sum()
    - 2000 * (is_decision & is_ng).sum()
)

part_auc = max(roc_auc - 0.5, 0) / 0.5
part_profit = max(total_net_profit, 0) / 20000

total_score = np.sqrt(part_auc * part_profit)

print(f"ROC-AUC Score        : {roc_auc:.6f}")
print(f"Total Net Profit     : {total_net_profit}")
print(f"Final Total Score    : {total_score:.6f}")

submission = pd.read_csv("../data/submission/sample_submission.csv")
submission['probability'] = np.concatenate([pred,pred])

decision_id_L_list = submission.iloc[:466].sort_values('probability').iloc[:200]['ID']
decision_id_P_list = submission.iloc[466:].sort_values('probability').iloc[:200]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True


submission.to_csv("../data/submission/my_submission_test_v2.csv", index=False)
# display(submission)