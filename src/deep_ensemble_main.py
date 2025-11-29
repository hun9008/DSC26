import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from CNN_encoder import (
    DataProcessor,
    SpatialRasterizer,
    FeatureEncoder,
    MultiModalDataset,
)

from util.eval import (
    evaluate_score_general,          # 안 쓰더라도 import 유지
    calculate_competition_score,
)

from util.logger import TeeLogger


# ----------------------------------------------------
# 0. Utility: 시드 고정
# ----------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------
# 1. Focal Loss (pos 클래스에 더 가중치)
# ----------------------------------------------------
class FocalLoss(nn.Module):
    """
    Binary Focal Loss (logits 입력)
    - alpha > 0.5 로 두면 양성 클래스(NG)에 더 큰 가중치
    - gamma 로 easy example down-weight
    """
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (N, 1) or (N,)
        targets: (N, 1) or (N,), float tensor in {0, 1}
        """
        if logits.dim() > 1:
            logits = logits.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)

        # BCE with logits (per-sample)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t: 예측 확률 (정답 클래스 기준)
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # alpha_t: 클래스별 가중치
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # focal factor
        focal_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_t * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ----------------------------------------------------
# 2. Deep Ensemble용 MLP
# ----------------------------------------------------
class EmbMLP(nn.Module):
    """
    입력: Encoder에서 뽑은 96차원 feature
    구조는 살짝 가볍게 (hidden_dim=64) 조정해서 과적합 완화
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits


# ----------------------------------------------------
# 3. Tabular Dataset (96차원 feature용)
# ----------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None
        if y is not None:
            self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]


# ----------------------------------------------------
# 4. 모델 하나 학습 + val/test 예측 반환
# ----------------------------------------------------
def train_one_ensemble_member(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    n_epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha_focal: float = 0.8,
    gamma_focal: float = 2.0,
    early_stopping_patience: int = 10,
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)

    best_state = None
    best_score = -1e9
    no_improve_cnt = 0

    # val용 y_true (고정)
    all_val_targets = []
    for _, y in val_loader:
        all_val_targets.append(y.numpy())
    all_val_targets = np.concatenate(all_val_targets, axis=0)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batch = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batch += 1

        avg_train_loss = train_loss_sum / max(n_batch, 1)

        # --- Validation ---
        model.eval()
        val_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
        val_probs = np.concatenate(val_probs, axis=0)

        # 현재 epoch 기준 score (k는 util.eval 내부 기본 사용)
        roc, profit, score = calculate_competition_score(
            y_true=all_val_targets,
            y_prob=val_probs,
        )

        print(
            f"[TrainOne] Epoch {epoch:03d} | "
            f"TrainLoss={avg_train_loss:.4f} | "
            f"ROC={roc:.6f} Profit={profit:.2f} Score={score:.6f}"
        )

        if score > best_score:
            best_score = score
            no_improve_cnt = 0
            best_state = model.state_dict()
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= early_stopping_patience:
                print(f"[TrainOne] Early stopping at epoch {epoch} (no improve {early_stopping_patience})")
                break

    # best state 로 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    # 최종 best 모델 기준 val prob / test prob 계산은
    # run_production_pipeline 내에서 다시 호출 (fold 안에서 처리)
    return model


# ----------------------------------------------------
# 5. Stratified K-Fold splits
# ----------------------------------------------------
def make_cv_splits(y_series, n_splits=5, base_seed=42):
    y = y_series.values
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=base_seed,
    )
    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        splits.append((train_idx, val_idx))
    return splits


# ----------------------------------------------------
# 6. OOF 기반 k_best 탐색
# ----------------------------------------------------
def find_best_k_by_oof(y_true, y_prob, k_min=5, k_max=120):
    """
    y_true : (N,) 0/1 numpy array
    y_prob : (N,) 예측 확률
    k_min, k_max : 탐색 범위 (대회 스펙에 맞춰 조정 가능)
    """
    best_k = None
    best_score = -1e9
    best_profit = None
    best_roc = None

    print("\n[BestK] Start sweeping k on OOF...")
    for k in range(k_min, k_max + 1):
        roc, profit, score = calculate_competition_score(
            y_true=y_true,
            y_prob=y_prob,
            k=k,  # util.eval에서 k 인자를 받는다고 가정
        )
        # 너무 로그가 많으면 이 부분 주석 처리 가능
        print(f"[BestK] k={k:3d} | ROC={roc:.6f} Profit={profit:.2f} Score={score:.6f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_profit = profit
            best_roc = roc

    print("\n[BestK] OOF best k summary")
    print(f"  k_best     : {best_k}")
    print(f"  ROC@k_best : {best_roc:.6f}")
    print(f"  Profit@k   : {best_profit:.2f}")
    print(f"  Score@k    : {best_score:.6f}")
    return best_k, best_roc, best_profit, best_score


# ----------------------------------------------------
# 7. 전체 Deep Ensemble 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """
    (사전 학습된) FeatureEncoder + Deep Ensemble(EmbMLP K개) 파이프라인
    - Encoder로 96차원 feature 추출
    - 각 fold마다 EmbMLP K개 학습 (seed 다르게)
    - fold 내 평균 → OOF / Test 예측
    - OOF 기반 k_best sweep
    """

    def __init__(
        self,
        n_epochs=60,
        batch_size=64,
        n_cv_splits=5,
        encoder_weight_path="../weight/feature_encoder.pth",
        ensemble_size=5,
    ):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Main] Device: {self.device}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path
        self.ensemble_size = ensemble_size

    # ---------------- Encoder 로드 ----------------
    def load_pretrained_encoder(self, basic_feature_dim):
        if not os.path.exists(self.encoder_weight_path):
            raise FileNotFoundError(
                f"Pretrained encoder weight not found: {self.encoder_weight_path}\n"
                f"먼저 encoder_train.py 를 실행해서 feature_encoder.pth 를 생성하세요."
            )

        self.feature_encoder = FeatureEncoder(
            basic_feature_dim=basic_feature_dim
        ).to(self.device)

        state_dict = torch.load(self.encoder_weight_path, map_location=self.device)
        self.feature_encoder.load_state_dict(state_dict)
        self.feature_encoder.eval()
        print(f"[Main] Loaded encoder weights from {self.encoder_weight_path}")

    # ---------------- Encoder feature 추출 ----------------
    def extract_features(self, loader, is_test=False):
        assert self.feature_encoder is not None, "feature_encoder가 로드되지 않았습니다."
        self.feature_encoder.eval()

        all_features = []
        with torch.no_grad():
            if is_test:
                for img, basic in loader:
                    img = img.to(self.device)
                    basic = basic.to(self.device)
                    feats = self.feature_encoder.extract_features(img, basic)
                    all_features.append(feats.cpu().numpy())
            else:
                for img, basic, _ in loader:
                    img = img.to(self.device)
                    basic = basic.to(self.device)
                    feats = self.feature_encoder.extract_features(img, basic)
                    all_features.append(feats.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    # ---------------- 전체 파이프라인 ----------------
    def run_production_pipeline(self):

        logger = TeeLogger()
        sys.stdout = logger

        print("[Main] Start Production Pipeline (Deep Ensemble)")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)

        # 4. 기본 피처 전처리 (Encoder용)
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"[Main] Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"[Main] Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. Dataset / DataLoader (Encoder feature 추출용)
        train_dataset_enc = MultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        test_dataset_enc = MultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )

        train_loader_enc = DataLoader(train_dataset_enc, batch_size=self.batch_size, shuffle=False)
        test_loader_enc = DataLoader(test_dataset_enc, batch_size=self.batch_size, shuffle=False)

        # 6. 사전 학습된 FeatureEncoder 로드
        self.load_pretrained_encoder(self.data_processor.basic_feature_dim)

        # 7. FeatureEncoder를 이용해 feature 추출 (96차원)
        X_train_feat = self.extract_features(train_loader_enc, is_test=False)
        X_test_feat = self.extract_features(test_loader_enc, is_test=True)

        print(f"[Main] Encoded features (Train): {X_train_feat.shape}")
        print(f"[Main] Encoded features (Test) : {X_test_feat.shape}")

        # 8. Hybrid features 설정 (지금은 encoder feature만 사용)
        X_train_hybrid = X_train_feat  # shape: (720, 96)
        X_test_hybrid = X_test_feat    # shape: (466, 96)

        print(f"[Main] Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"[Main] Hybrid features (Test) : {X_test_hybrid.shape}")

        # 9. CV splits
        cv_splits = make_cv_splits(
            train_Y_series,
            n_splits=self.n_cv_splits,
            base_seed=42
        )

        y_all = train_Y_series.values
        n_train = len(y_all)
        n_test = X_test_hybrid.shape[0]

        approx_val_size = n_train // self.n_cv_splits
        print(f"\n[Main] Cross Validation with {self.n_cv_splits} folds "
              f"(approx each val size: {approx_val_size})")
        print(f"(Deep Ensemble with {self.ensemble_size} members per fold)")

        # OOF / Test 예측 저장
        oof_preds = np.zeros(n_train, dtype=np.float32)
        test_preds_folds = []  # 각 fold에서 (n_test,) ensemble 평균

        # ------------------- CV Loop -------------------
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")
            X_tr = X_train_hybrid[train_idx]
            y_tr = y_all[train_idx]
            X_val = X_train_hybrid[val_idx]
            y_val = y_all[val_idx]

            print(f"[Main] Train size: {X_tr.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} (NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            # Dataset / Loader 준비
            train_ds = TabularDataset(X_tr, y_tr)
            val_ds = TabularDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

            val_probs_members = []
            test_probs_members = []

            # Deep Ensemble: 동일 구조 MLP를 K개 학습
            for m in range(self.ensemble_size):
                print(f"[Fold {fold_idx+1}] Train ensemble member {m+1}/{self.ensemble_size}")
                # seed를 fold와 member 기반으로 다르게
                set_global_seed(1000 * (fold_idx + 1) + m + 1)

                model = EmbMLP(input_dim=X_train_hybrid.shape[1], hidden_dim=64, dropout=0.3)
                model = train_one_ensemble_member(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                    n_epochs=self.n_epochs,
                    lr=1e-3,
                    weight_decay=1e-4,
                    alpha_focal=0.8,   # 양성(NG)에 더 큰 가중치 → pos_weight 비슷한 효과
                    gamma_focal=2.0,
                    early_stopping_patience=10,
                )

                # best 모델 기준 fold val/test prob 계산
                model.eval()
                with torch.no_grad():
                    # Val
                    val_probs = []
                    for xb, _ in val_loader:
                        xb = xb.to(self.device)
                        logits = model(xb).view(-1)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        val_probs.append(probs)
                    val_probs = np.concatenate(val_probs, axis=0)
                    val_probs_members.append(val_probs)

                    # Test
                    test_ds = TabularDataset(X_test_hybrid, y=None)
                    test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
                    test_probs = []
                    for xb in test_loader:
                        xb = xb.to(self.device)
                        logits = model(xb).view(-1)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        test_probs.append(probs)
                    test_probs = np.concatenate(test_probs, axis=0)
                    test_probs_members.append(test_probs)

            # Fold 내 ensemble 평균
            val_probs_mean = np.mean(val_probs_members, axis=0)   # (val,)
            test_probs_mean = np.mean(test_probs_members, axis=0) # (test,)

            # OOF 채우기
            oof_preds[val_idx] = val_probs_mean
            test_preds_folds.append(test_probs_mean)

            # Fold 성능 (기본 calculate_competition_score 기준)
            roc_f, profit_f, score_f = calculate_competition_score(
                y_true=y_val,
                y_prob=val_probs_mean,
            )
            print(f"[Fold {fold_idx+1}] DeepEns ROC={roc_f:.6f} Profit={profit_f:.2f} Score={score_f:.6f}")

        # ------------------- OOF 성능 (기본) -------------------
        print("\n===== OOF Performance (Deep Ensemble, default k behavior) =====")
        roc_oof, profit_oof, score_oof = calculate_competition_score(
            y_true=y_all,
            y_prob=oof_preds,
        )
        print(f"ROC-AUC = {roc_oof:.6f}, Profit = {profit_oof:.2f}, Score = {score_oof:.6f}")

        # ------------------- OOF 기반 k_best 탐색 -------------------
        k_best, roc_k, profit_k, score_k = find_best_k_by_oof(
            y_true=y_all,
            y_prob=oof_preds,
            k_min=5,
            k_max=120,
        )

        # Fold별로도 k_best 기준으로 다시 요약 (선택사항)
        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []
        for fold_idx, (_, val_idx) in enumerate(cv_splits):
            y_val = y_all[val_idx]
            val_prob = oof_preds[val_idx]
            roc_f, profit_f, score_f = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob,
                k=k_best,
            )
            cv_roc_list.append(roc_f)
            cv_profit_list.append(profit_f)
            cv_score_list.append(score_f)
            print(f"[Fold {fold_idx+1}] (k_best={k_best}) ROC={roc_f:.6f} Profit={profit_f:.2f} Score={score_f:.6f}")

        print("\n===== CV Summary (Deep Ensemble @ k_best) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # ------------------- Test 예측 (fold 평균) -------------------
        test_preds_mean = np.mean(test_preds_folds, axis=0)  # (n_test,)
        print(f"\n[Main] Test prob range: {test_preds_mean.min():.4f} ~ {test_preds_mean.max():.4f}")

        # ------------------- 제출 파일 생성 (기존 로직 유지) -------------------
        submission = pd.read_csv("../data/submission/sample_submission.csv")
        # L/P 각각 동일 확률 사용
        submission['probability'] = np.concatenate([test_preds_mean, test_preds_mean])
        submission['decision'] = False

        n_sub = len(submission)
        half_sub = n_sub // 2
        idx_L_sub = submission.index[:half_sub]
        idx_P_sub = submission.index[half_sub:]

        # 여기서는 기존 규칙 유지 (L/P 각각 170개 선택)
        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:200]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:200]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/DeepEns_Focal_OOFk_{timestamp}.csv"
        submission.to_csv(save_path, index=False)
        print(f"[Main] Saved submission to {save_path}")

        selected_count = submission['decision'].sum()
        print(f"[Main] Total selected products: {selected_count}")
        print(f"[Main] OOF k_best = {k_best}, Score@k_best = {score_k:.6f}")

        logger.close()
        sys.stdout = sys.__stdout__
        print(f"[Main] Log saved to: {logger.log_path}")

        return submission


# ----------------------------------------------------
# 8. main()
# ----------------------------------------------------
def main():
    pipeline = ProductionPipeline(
        n_epochs=60,                     # epoch 살짝 줄임 (과적합 완화)
        batch_size=64,
        n_cv_splits=5,
        encoder_weight_path="../weight/feature_encoder.pth",
        ensemble_size=5,                # Deep Ensemble 멤버 수
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()