import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

from util.logger import TeeLogger
import sys

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
submission = pd.read_csv("../data/submission/sample_submission.csv")

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

def preprocess(dataset: pd.DataFrame) -> np.ndarray:
    Xc = OE.transform(dataset[cat_list])
    Xn = np.array(dataset[num_list])
    return np.concatenate([Xc, Xn], axis=1)

logger = TeeLogger()
sys.stdout = logger

model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    random_state=42
)
model.fit(preprocess(train_X), train_Y)

pred = model.predict_proba(preprocess(test_X))[:, 1]

train_pred = model.predict_proba(preprocess(train_X))[:, 1]


roc_auc, total_net_profit, total_score = evaluate_score_general(
    y_ng=train_Y.values,
    prob_ng=train_pred,
    n_select_each=200,
    profit_good=100,
    cost_ng=2000,
)

print(f"ROC-AUC Score        : {roc_auc:.6f}")
print(f"Total Net Profit     : {total_net_profit}")
print(f"Final Total Score    : {total_score:.6f}")

submission = pd.read_csv("../data/submission/sample_submission.csv")
submission['probability'] = np.concatenate([pred, pred])

submission['decision'] = False

n_sub = len(submission)
half = n_sub // 2

idx_L = submission.index[:half]
idx_P = submission.index[half:]

decision_id_L_list = submission.loc[idx_L].sort_values(
    'probability', ascending=True
).iloc[:200]['ID']

decision_id_P_list = submission.loc[idx_P].sort_values(
    'probability', ascending=True
).iloc[:200]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

submission.to_csv("../data/submission/my_submission.csv", index=False)
# display(submission)  

logger.close()
sys.stdout = sys.__stdout__
print(f"[Main] Log saved to: {logger.log_path}")