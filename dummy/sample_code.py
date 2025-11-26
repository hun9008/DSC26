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
    cost_ng=2000
):
    y_ng = np.asarray(y_ng)
    prob_ng = np.asarray(prob_ng)
    n = len(y_ng)
    assert len(prob_ng) == n

    y_good = 1 - y_ng
    prob_good = 1.0 - prob_ng

    eval_df = pd.DataFrame({
        "y_good": y_good,
        "prob_ng": prob_ng,
        "prob_good": prob_good
    })

    half = n // 2
    eval_df["decision"] = False

    top_L = eval_df.iloc[:half].sort_values("prob_ng").iloc[:n_select_each].index
    top_P = eval_df.iloc[half:].sort_values("prob_ng").iloc[:n_select_each].index

    eval_df.loc[top_L, "decision"] = True
    eval_df.loc[top_P, "decision"] = True

    roc_auc = roc_auc_score(eval_df["y_good"], eval_df["prob_good"])

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


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/sample_submission.csv")

train_X = train.drop(columns=['Class']).iloc[:, :-256*3]
train_Y_ng = (train['Class'] == 'NG').astype(int)
test_X = test.drop(columns=['ID']).iloc[:, :-256*3]

cat_list = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)))

OE = OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist', sparse_output=False)
if len(cat_list) > 0:
    OE.fit(train_X[cat_list])
else:
    OE.fit(pd.DataFrame(index=train_X.index))

def preprocess(dataset):
    if len(cat_list) > 0:
        Xc = OE.transform(dataset[cat_list])
    else:
        Xc = np.zeros((len(dataset), 0))
    Xn = np.array(dataset[num_list])
    return np.concatenate([Xc, Xn], axis=1)

model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)
model.fit(preprocess(train_X), train_Y_ng)

train_prob_ng = model.predict_proba(preprocess(train_X))[:, 1]
pred_prob_ng = model.predict_proba(preprocess(test_X))[:, 1]

roc_auc, total_net_profit, total_score = evaluate_score_general(
    y_ng=train_Y_ng,
    prob_ng=train_prob_ng,
    n_select_each=200,
    profit_good=100,
    cost_ng=2000
)

submission['probability'] = np.concatenate([pred_prob_ng, pred_prob_ng])
submission['decision'] = False

n_sub = len(submission)
half_sub = n_sub // 2

idx_L_sub = submission.index[:half_sub]
idx_P_sub = submission.index[half_sub:]

decision_id_L_list = submission.loc[idx_L_sub].sort_values('probability', ascending=True).iloc[:200]['ID']
decision_id_P_list = submission.loc[idx_P_sub].sort_values('probability', ascending=True).iloc[:200]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

submission.to_csv("./data/my_submission.csv", index=False)