import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_auc_score
from datetime import datetime

from util.eval_v2 import (
    eval_official_on_probs
)

from util.logger import TeeLogger
import sys

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

logger = TeeLogger()
sys.stdout = logger

model = RandomForestClassifier(n_estimators=1000, max_depth=5)
model.fit(preprocess(train_X), train_Y)

pred = model.predict_proba(preprocess(test_X))[:,1]

train_pred = model.predict_proba(preprocess(train_X))[:, 1]

roc_auc, total_net_profit, total_score = eval_official_on_probs(
    y_ng=train_Y.values,
    prob_ng=train_pred,
)

print(f"ROC-AUC Score        : {roc_auc:.6f}")
print(f"Total Net Profit     : {total_net_profit}")
print(f"Final Total Score    : {total_score:.6f}")

submission = pd.read_csv("../data/submission/sample_submission.csv")
submission['probability'] = np.concatenate([pred,pred])

decision_id_L_list = submission.iloc[:466].sort_values('probability').iloc[:200]['ID']
decision_id_P_list = submission.iloc[466:].sort_values('probability').iloc[:200]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"../data/submission/my_submission_{timestamp}.csv"

submission.to_csv(save_path, index=False)
print(f"[Main] Saved submission to {save_path}")

logger.close()
sys.stdout = sys.__stdout__
print(f"[Main] Log saved to: {logger.log_path}")