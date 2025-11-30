import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from scipy.stats import rankdata

from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)
from util.logger import TeeLogger

# --- GBDT 계열 ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- NN용 ---
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

os.environ["OMP_NUM_THREADS"] = "1"


# ==============================
# 1. 시퀀스 기반 통계 피처 생성
# ==============================
def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    x0~x255, y0~y255, p0~p255 시퀀스에서 통계 피처를 생성하고
    원래 시퀀스 컬럼은 드롭한 새로운 DataFrame을 반환.
    """
    df = df.copy()

    # 시퀀스 컬럼 탐색
    x_cols = [f"x{i}" for i in range(256) if f"x{i}" in df.columns]
    y_cols = [f"y{i}" for i in range(256) if f"y{i}" in df.columns]
    p_cols = [f"p{i}" for i in range(256) if f"p{i}" in df.columns]

    if not (x_cols and y_cols and p_cols):
        # 시퀀스가 없다면 그대로 반환
        return df

    x_arr = df[x_cols].values.astype(float)
    y_arr = df[y_cols].values.astype(float)
    p_arr = df[p_cols].values.astype(float)

    def make_seq_stats(prefix: str, arr: np.ndarray) -> dict:
        stats = {}
        # 전역 통계
        stats[f"{prefix}_mean"] = arr.mean(axis=1)
        stats[f"{prefix}_std"] = arr.std(axis=1)
        stats[f"{prefix}_min"] = arr.min(axis=1)
        stats[f"{prefix}_max"] = arr.max(axis=1)
        stats[f"{prefix}_range"] = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]

        q25 = np.percentile(arr, 25, axis=1)
        q50 = np.percentile(arr, 50, axis=1)
        q75 = np.percentile(arr, 75, axis=1)
        stats[f"{prefix}_q25"] = q25
        stats[f"{prefix}_q50"] = q50
        stats[f"{prefix}_q75"] = q75

        # 4 구간 나눠서 구간별 mean/std
        chunks = np.array_split(arr, 4, axis=1)
        for i, chunk in enumerate(chunks):
            stats[f"{prefix}_seg{i}_mean"] = chunk.mean(axis=1)
            stats[f"{prefix}_seg{i}_std"] = chunk.std(axis=1)

        # 1-step 차분 기반 통계
        d = np.diff(arr, axis=1)
        ad = np.abs(d)
        stats[f"{prefix}_d_mean_abs"] = ad.mean(axis=1)
        stats[f"{prefix}_d_std_abs"] = ad.std(axis=1)
        stats[f"{prefix}_d_max_abs"] = ad.max(axis=1)

        # 큰 점프(예: 95% 이상) 카운트
        thr = np.percentile(ad, 95, axis=1)
        big_jump = (ad >= thr[:, None]).sum(axis=1)
        stats[f"{prefix}_big_jump_cnt"] = big_jump

        return stats

    # x, y, p 각각에 대한 통계 피처
    feat_dict = {}
    feat_dict.update(make_seq_stats("xseq", x_arr))
    feat_dict.update(make_seq_stats("yseq", y_arr))
    feat_dict.update(make_seq_stats("pseq", p_arr))

    # x-y 궤적 길이(폴리라인 길이)
    dx = np.diff(x_arr, axis=1)
    dy = np.diff(y_arr, axis=1)
    path = np.sqrt(dx ** 2 + dy ** 2).sum(axis=1)
    feat_dict["xy_path_length"] = path

    seq_feat_df = pd.DataFrame(feat_dict, index=df.index)

    # 원래 시퀀스 컬럼 제거 후 통계 피처 붙이기
    seq_cols = x_cols + y_cols + p_cols
    df_no_seq = df.drop(columns=seq_cols)

    df_new = pd.concat([df_no_seq, seq_feat_df], axis=1)
    return df_new


# ==============================
# 2. 다운샘플링 인덱스 생성 공통 함수
# ==============================
def make_downsample_indices(y: np.ndarray,
                            neg_pos_ratio: float,
                            rng: np.random.RandomState):
    """
    NG(1) 전부 + Good(0)에서 neg_pos_ratio * N_pos 만큼 다운샘플.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg_sample = min(len(neg_idx), int(neg_pos_ratio * n_pos))

    sampled_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
    train_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(train_idx)
    return train_idx


# ==============================
# 3. 모델별 bagging 학습 함수들
# ==============================
def train_bagged_rf(X, y, n_bags=10, neg_pos_ratio=2.0, random_state=42):
    models = []
    rng = np.random.RandomState(random_state)
    for b in range(n_bags):
        train_idx = make_downsample_indices(y, neg_pos_ratio, rng)
        X_tr, y_tr = X[train_idx], y[train_idx]

        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            n_jobs=-1,
            random_state=random_state + b,
        )
        clf.fit(X_tr, y_tr)
        models.append(clf)
    return models


def train_bagged_lgbm(X, y, n_bags=10, neg_pos_ratio=2.0, random_state=142):
    models = []
    rng = np.random.RandomState(random_state)
    for b in range(n_bags):
        train_idx = make_downsample_indices(y, neg_pos_ratio, rng)
        X_tr, y_tr = X[train_idx], y[train_idx]

        clf = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=random_state + b,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        models.append(clf)
    return models


def train_bagged_xgb(X, y, n_bags=10, neg_pos_ratio=2.0, random_state=242):
    models = []
    rng = np.random.RandomState(random_state)
    for b in range(n_bags):
        train_idx = make_downsample_indices(y, neg_pos_ratio, rng)
        X_tr, y_tr = X[train_idx], y[train_idx]

        clf = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
            random_state=random_state + b,
        )
        clf.fit(X_tr, y_tr)
        models.append(clf)
    return models


def train_bagged_catboost(X, y, n_bags=10, neg_pos_ratio=2.0, random_state=342):
    models = []
    rng = np.random.RandomState(random_state)
    for b in range(n_bags):
        train_idx = make_downsample_indices(y, neg_pos_ratio, rng)
        X_tr, y_tr = X[train_idx], y[train_idx]

        clf = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state + b,
            verbose=False,
            thread_count=-1,
        )
        clf.fit(X_tr, y_tr)
        models.append(clf)
    return models


# ==============================
# 4. NN(MLP) 모델 & bagging
# ==============================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def train_single_mlp(X_tr, y_tr, input_dim, device, seed=0,
                     n_epochs=40, batch_size=64, lr=1e-3):
    torch.manual_seed(seed)
    model = SimpleMLP(input_dim).to(device)

    ds = TensorDataset(
        torch.from_numpy(X_tr.astype(np.float32)),
        torch.from_numpy(y_tr.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


def train_bagged_mlp(X, y, n_bags=5, neg_pos_ratio=2.0, random_state=442, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    rng = np.random.RandomState(random_state)
    input_dim = X.shape[1]

    for b in range(n_bags):
        train_idx = make_downsample_indices(y, neg_pos_ratio, rng)
        X_tr, y_tr = X[train_idx], y[train_idx]

        model = train_single_mlp(
            X_tr, y_tr, input_dim=input_dim,
            device=device,
            seed=random_state + b,
            n_epochs=40,
            batch_size=64,
            lr=1e-3,
        )
        models.append(model)
    return models


# ==============================
# 5. 모델 리스트에서 예측 평균
# ==============================
def predict_bagged_sklearn(models, X: np.ndarray) -> np.ndarray:
    preds = np.zeros(X.shape[0], dtype=float)
    for m in models:
        preds += m.predict_proba(X)[:, 1]
    preds /= len(models)
    return preds


def predict_bagged_mlp(models, X: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    preds = np.zeros(X.shape[0], dtype=float)

    for m in models:
        m.eval()
        with torch.no_grad():
            logits = m(X_t).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
        preds += probs
    preds /= len(models)
    return preds


# ==============================
# 6. rank-based weighted 앙상블
# ==============================
def rank_based_ensemble(pred_dict: dict, weights: dict) -> np.ndarray:
    """
    pred_dict: {model_name: np.array(shape=(N,))}
    weights  : {model_name: weight}

    각 모델의 예측을 rank로 변환 후 [0,1] 스케일로 normalize한 뒤
    가중합 / 가중합합 으로 최종 score 생성.
    """
    assert set(pred_dict.keys()) == set(weights.keys())

    n = len(next(iter(pred_dict.values())))
    eps = 1e-9

    combined = np.zeros(n, dtype=float)
    w_sum = 0.0

    for name, preds in pred_dict.items():
        # rank: 값이 클수록 NG 확률이 크다고 가정
        r = rankdata(preds, method="average")  # 1 ~ N
        r = (r - 1) / max(1, (n - 1))  # 0 ~ 1
        w = float(weights[name])
        combined += w * r
        w_sum += w

    combined /= (w_sum + eps)
    return combined


# ==============================
# 7. 메인 파이프라인
# ==============================
def main():
    logger = TeeLogger()
    sys.stdout = logger

    print("[Main] Load data")
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    submission_template = pd.read_csv("../data/submission/sample_submission.csv")

    # -------------------------
    # 7-1. 시퀀스 통계 피처 추가
    # -------------------------
    print("[Main] Add sequence-based statistical features")
    train_feat = add_sequence_features(train)
    test_feat = add_sequence_features(test)

    # 타깃 분리
    y = train_feat["Class"].map({"NG": 1, "Good": 0}).values

    # ID / Class 제거한 feature DataFrame
    drop_cols_train = [c for c in ["ID", "Class"] if c in train_feat.columns]
    drop_cols_test = [c for c in ["ID"] if c in test_feat.columns]

    X_train_df = train_feat.drop(columns=drop_cols_train)
    X_test_df = test_feat.drop(columns=drop_cols_test)

    # -------------------------
    # 7-2. 범주형 / 수치형 컬럼 분리 + OneHot
    # -------------------------
    print("[Main] Setup categorical & numeric preprocessing")

    cat_list = X_train_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    num_list = sorted(list(set(X_train_df.columns) - set(cat_list)))

    print(f"[Main] Categorical features: {cat_list}")
    print(f"[Main] Numeric features    : {len(num_list)} columns")

    OE = OneHotEncoder(
        min_frequency=0.01,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    if cat_list:
        OE.fit(X_train_df[cat_list])
    else:
        OE = None

    def preprocess(df: pd.DataFrame) -> np.ndarray:
        if cat_list:
            Xc = OE.transform(df[cat_list])
        else:
            Xc = np.zeros((len(df), 0))
        Xn = df[num_list].to_numpy(dtype=float) if num_list else np.zeros((len(df), 0))
        return np.concatenate([Xc, Xn], axis=1)

    X_train = preprocess(X_train_df)
    X_test = preprocess(X_test_df)

    print(f"[Main] X_train shape: {X_train.shape}")
    print(f"[Main] X_test  shape: {X_test.shape}")
    print(f"[Main] Target NG count: {(y == 1).sum()}, Good count: {(y == 0).sum()}")

    # -------------------------
    # 7-3. Stratified K-Fold OOF 평가 + rank 앙상블
    # -------------------------
    N_SPLITS = 5
    NEG_POS_RATIO = 2.0

    # 각 모델 타입별 OOF prediction 저장
    oof_rf = np.zeros(len(y), dtype=float)
    oof_lgbm = np.zeros(len(y), dtype=float)
    oof_xgb = np.zeros(len(y), dtype=float)
    oof_cat = np.zeros(len(y), dtype=float)
    oof_mlp = np.zeros(len(y), dtype=float)
    oof_ens = np.zeros(len(y), dtype=float)

    # rank-based weighted ensemble weight
    # (LGBM/XGB/CatBoost를 조금 더 신뢰한다고 가정)
    ens_weights = {
        "rf": 1.0,
        "lgbm": 1.5,
        "xgb": 1.5,
        "cat": 1.5,
        "mlp": 1.0,
    }

    print(
        f"[Main] Start StratifiedKFold CV (n_splits={N_SPLITS}, "
        f"neg_pos_ratio={NEG_POS_RATIO})"
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] NN device: {device}")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n[CV] Fold {fold + 1}/{N_SPLITS}")
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_val, y_val = X_train[val_idx], y[val_idx]

        print(
            f"[CV] Train size: {len(tr_idx)}, "
            f"Val size: {len(val_idx)}, "
            f"NG in val: {(y_val == 1).sum()}, Good in val: {(y_val == 0).sum()}"
        )

        # --- 각 모델 타입별 bagging 학습 ---
        rf_models = train_bagged_rf(
            X_tr, y_tr, n_bags=10, neg_pos_ratio=NEG_POS_RATIO,
            random_state=42 + fold
        )
        lgbm_models = train_bagged_lgbm(
            X_tr, y_tr, n_bags=10, neg_pos_ratio=NEG_POS_RATIO,
            random_state=142 + fold
        )
        xgb_models = train_bagged_xgb(
            X_tr, y_tr, n_bags=10, neg_pos_ratio=NEG_POS_RATIO,
            random_state=242 + fold
        )
        cat_models = train_bagged_catboost(
            X_tr, y_tr, n_bags=10, neg_pos_ratio=NEG_POS_RATIO,
            random_state=342 + fold
        )
        mlp_models = train_bagged_mlp(
            X_tr, y_tr, n_bags=5, neg_pos_ratio=NEG_POS_RATIO,
            random_state=442 + fold,
            device=device,
        )

        # --- 각 모델 타입별 validation 예측 ---
        val_rf = predict_bagged_sklearn(rf_models, X_val)
        val_lgbm = predict_bagged_sklearn(lgbm_models, X_val)
        val_xgb = predict_bagged_sklearn(xgb_models, X_val)
        val_cat = predict_bagged_sklearn(cat_models, X_val)
        val_mlp = predict_bagged_mlp(mlp_models, X_val, device=device)

        oof_rf[val_idx] = val_rf
        oof_lgbm[val_idx] = val_lgbm
        oof_xgb[val_idx] = val_xgb
        oof_cat[val_idx] = val_cat
        oof_mlp[val_idx] = val_mlp

        # --- rank-based weighted ensemble ---
        val_pred_dict = {
            "rf": val_rf,
            "lgbm": val_lgbm,
            "xgb": val_xgb,
            "cat": val_cat,
            "mlp": val_mlp,
        }
        val_ens = rank_based_ensemble(val_pred_dict, ens_weights)
        oof_ens[val_idx] = val_ens

        # fold별 ensemble 점수 출력
        roc_auc, total_net_profit, total_score = calculate_competition_score(
            y_true=y_val,
            y_prob=val_ens,
        )
        print(f"[CV] Fold {fold + 1} Ensemble ROC-AUC : {roc_auc:.6f}")
        print(f"[CV] Fold {fold + 1} Ensemble Profit  : {total_net_profit}")
        print(f"[CV] Fold {fold + 1} Ensemble Score   : {total_score:.6f}")

    # -------------------------
    # 7-4. 전체 OOF 기준 점수 (ensemble)
    # -------------------------
    print("\n[Main] OOF evaluation over all folds (rank-based ensemble):")
    roc_auc, total_net_profit, total_score = calculate_competition_score(
        y_true=y,
        y_prob=oof_ens,
    )
    print(f"OOF Ensemble ROC-AUC Score        : {roc_auc:.6f}")
    print(f"OOF Ensemble Total Net Profit     : {total_net_profit}")
    print(f"OOF Ensemble Final Total Score    : {total_score:.6f}")

    # (옵션) 개별 모델 OOF도 보고 싶으면 아래 주석 해제
    for name, oof_pred in {
        "rf": oof_rf,
        "lgbm": oof_lgbm,
        "xgb": oof_xgb,
        "cat": oof_cat,
        "mlp": oof_mlp,
    }.items():
        roc_auc, total_net_profit, total_score = calculate_competition_score(
            y_true=y,
            y_prob=oof_pred,
        )
        print(f"[OOF-{name}] ROC-AUC={roc_auc:.6f}, Profit={total_net_profit}, Score={total_score:.6f}")

    # -------------------------
    # 7-5. 전체 train으로 최종 모델 학습 → test 예측
    # -------------------------
    print("\n[Main] Train final bagged models on full training data")

    final_rf_models = train_bagged_rf(
        X_train, y, n_bags=10, neg_pos_ratio=NEG_POS_RATIO, random_state=1001
    )
    final_lgbm_models = train_bagged_lgbm(
        X_train, y, n_bags=10, neg_pos_ratio=NEG_POS_RATIO, random_state=1101
    )
    final_xgb_models = train_bagged_xgb(
        X_train, y, n_bags=10, neg_pos_ratio=NEG_POS_RATIO, random_state=1201
    )
    final_cat_models = train_bagged_catboost(
        X_train, y, n_bags=10, neg_pos_ratio=NEG_POS_RATIO, random_state=1301
    )
    final_mlp_models = train_bagged_mlp(
        X_train, y, n_bags=5, neg_pos_ratio=NEG_POS_RATIO, random_state=1401, device=device
    )

    test_rf = predict_bagged_sklearn(final_rf_models, X_test)
    test_lgbm = predict_bagged_sklearn(final_lgbm_models, X_test)
    test_xgb = predict_bagged_sklearn(final_xgb_models, X_test)
    test_cat = predict_bagged_sklearn(final_cat_models, X_test)
    test_mlp = predict_bagged_mlp(final_mlp_models, X_test, device=device)

    print(
        f"[Main] Test RF  prob range: {test_rf.min():.6f} ~ {test_rf.max():.6f}"
    )
    print(
        f"[Main] Test LGBM prob range: {test_lgbm.min():.6f} ~ {test_lgbm.max():.6f}"
    )
    print(
        f"[Main] Test XGB prob range : {test_xgb.min():.6f} ~ {test_xgb.max():.6f}"
    )
    print(
        f"[Main] Test CAT prob range : {test_cat.min():.6f} ~ {test_cat.max():.6f}"
    )
    print(
        f"[Main] Test MLP prob range : {test_mlp.min():.6f} ~ {test_mlp.max():.6f}"
    )

    test_pred_dict = {
        "rf": test_rf,
        "lgbm": test_lgbm,
        "xgb": test_xgb,
        "cat": test_cat,
        "mlp": test_mlp,
    }
    test_ens = rank_based_ensemble(test_pred_dict, ens_weights)
    print(
        f"[Main] Test Ensemble score range (rank-combined): "
        f"{test_ens.min():.6f} ~ {test_ens.max():.6f}"
    )

    # -------------------------
    # 7-6. 제출 파일 생성 (L/P 각각 하위 k개 선택)
    # -------------------------
    submission = submission_template.copy()
    # rank-based ensemble output을 그대로 "probability"로 사용
    submission["probability"] = np.concatenate([test_ens, test_ens])

    if "decision" not in submission.columns:
        submission["decision"] = False
    else:
        submission["decision"] = False

    n_total = len(submission)
    half = n_total // 2
    n_select = 200  # 규칙에 맞게 조정

    idx_L = submission.index[:half]
    idx_P = submission.index[half:]

    decision_id_L_list = (
        submission.loc[idx_L]
        .sort_values("probability", ascending=True)
        .iloc[:n_select]["ID"]
    )
    decision_id_P_list = (
        submission.loc[idx_P]
        .sort_values("probability", ascending=True)
        .iloc[:n_select]["ID"]
    )

    submission.loc[submission["ID"].isin(decision_id_L_list), "decision"] = True
    submission.loc[submission["ID"].isin(decision_id_P_list), "decision"] = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"../data/submission/full_ensemble_rank_{timestamp}.csv"
    submission.to_csv(save_path, index=False)

    selected_count = submission["decision"].sum()
    print(f"[Main] Saved submission to {save_path}")
    print(f"[Main] Total selected products: {selected_count}")

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"[Main] Log saved to: {logger.log_path}")


if __name__ == "__main__":
    main()