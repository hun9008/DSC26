# main_pipeline_ae.py (예: 기존 파일 덮어쓰거나 새 이름으로 저장)

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from CNN_encoder import (
    DataProcessor,
    SpatialRasterizer,
    FeatureEncoder,
    MultiModalDataset,
)

# 평가 함수 공통 모듈
from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

from util.logger import TeeLogger


# ----------------------------------------------------
# 0. AutoEncoder 정의 (96 → 48 → 24 → 48 → 96)
# ----------------------------------------------------
class FeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=64, latent_dim=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ----------------------------------------------------
# 1. Main Model (RandomForest 기반 이진 분류기)
# ----------------------------------------------------
class MainModel:
    """
    Main Model:
      - 입력: AE까지 거친 최종 feature (hybrid + latent + recon_error)
      - 모델: RandomForestClassifier
      - 출력: NG 확률
    """

    def __init__(self, n_estimators=200, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 2. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """(사전 학습된) FeatureEncoder + AutoEncoder + RF 파이프라인"""

    def __init__(self, n_epochs=13, batch_size=32, n_cv_splits=5,
                 encoder_weight_path="feature_encoder.pth"):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None
        self.main_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path

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

    # ---------------- AutoEncoder 학습 ----------------
    def train_autoencoder(self,
                          X_train_feat_np,
                          n_epochs=200,
                          batch_size=64,
                          lr=1e-3,
                          weight_decay=1e-5):
        """
        X_train_feat_np : (N, D) = (720, 96) 정도
        return: 학습된 AE 모델
        """
        input_dim = X_train_feat_np.shape[1]
        ae = FeatureAutoEncoder(input_dim=input_dim,
                                hidden_dim=64,
                                latent_dim=24).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_train_feat_np.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        print(f"[AE] Train AutoEncoder: input_dim={input_dim}, hidden_dim=64, latent_dim=24")
        for ep in range(1, n_epochs + 1):
            ae.train()
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                optimizer.zero_grad()
                x_hat, z = ae(x_batch)
                loss = criterion(x_hat, x_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)

            avg_loss = total_loss / len(dataset)
            if ep % 20 == 0 or ep == 1:
                print(f"[AE] Epoch {ep}/{n_epochs} | Recon MSE={avg_loss:.6f}")

        return ae

    # ---------------- AE로 latent/recon 추출 ----------------
    def get_ae_features(self, ae_model, X_np):
        """
        ae_model : 학습된 AutoEncoder
        X_np     : (N, D) numpy (train or test)
        return   : latent (N, latent_dim), recon_error (N, 1)
        """
        ae_model.eval()
        all_latent = []
        all_recon_err = []

        dataset = TensorDataset(torch.from_numpy(X_np.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                x_hat, z = ae_model(x_batch)
                # 재구성 오차: sample-wise MSE, shape (batch, 1)
                recon_err = ((x_hat - x_batch) ** 2).mean(dim=1, keepdim=True)

                all_latent.append(z.cpu().numpy())
                all_recon_err.append(recon_err.cpu().numpy())

        latent_np = np.concatenate(all_latent, axis=0)
        recon_np = np.concatenate(all_recon_err, axis=0)
        return latent_np, recon_np

    # ---------------- Stratified K-Fold CV splits 생성 ----------------
    @staticmethod
    def make_cv_splits(y_series, n_splits=5, base_seed=42):
        """
        StratifiedKFold를 사용하여 label 비율을 유지한 채
        대략 len(y_series) / n_splits 개씩 validation을 만드는 함수.
        """
        y = y_series.values
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=base_seed
        )
        splits = []
        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            splits.append((train_idx, val_idx))
        return splits

    # ---------------- 전체 파이프라인 실행 ----------------
    def run_production_pipeline(self):

        logger = TeeLogger()
        sys.stdout = logger

        print("[Main] Start Production Pipeline (Encoder + AE + RF)")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)

        # 4. 기본 피처 전처리
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"[Main] Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"[Main] Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. Dataset / DataLoader (feature 추출용)
        train_dataset = MultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        test_dataset = MultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )

        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 6. 사전 학습된 FeatureEncoder 로드
        self.load_pretrained_encoder(self.data_processor.basic_feature_dim)

        # 7. FeatureEncoder를 이용해 feature 추출 (Z: 96차원)
        X_train_feat = self.extract_features(train_loader_seq, is_test=False)
        X_test_feat = self.extract_features(test_loader, is_test=True)

        print(f"[Main] Encoded features (Train): {X_train_feat.shape}")  # (720, 96)
        print(f"[Main] Encoded features (Test) : {X_test_feat.shape}")   # (466, 96)

        # 8. AutoEncoder 학습 (Z 위에서 비지도)
        ae = self.train_autoencoder(
            X_train_feat_np=X_train_feat,
            n_epochs=200,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-5,
        )

        # 9. AE latent / recon_error 추출
        z_train_latent, z_train_recon = self.get_ae_features(ae, X_train_feat)
        z_test_latent, z_test_recon = self.get_ae_features(ae, X_test_feat)

        print(f"[AE] Train latent shape      : {z_train_latent.shape}")   # (720, 24)
        print(f"[AE] Train recon_error shape : {z_train_recon.shape}")    # (720, 1)
        print(f"[AE] Test latent shape       : {z_test_latent.shape}")    # (466, 24)
        print(f"[AE] Test recon_error shape  : {z_test_recon.shape}")     # (466, 1)

        # 10. 최종 feature 구성: [Z, z_latent, recon_error]
        X_train_final = np.concatenate([X_train_feat, z_train_latent, z_train_recon], axis=1)
        X_test_final = np.concatenate([X_test_feat, z_test_latent, z_test_recon], axis=1)

        print(f"[Main] Final features (Train): {X_train_final.shape}")
        print(f"[Main] Final features (Test) : {X_test_final.shape}")

        # 11. Cross Validation (Stratified K-Fold)
        cv_splits = self.make_cv_splits(
            train_Y_series,
            n_splits=self.n_cv_splits,
            base_seed=42
        )

        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []

        approx_val_size = len(train_Y_series) // self.n_cv_splits
        print(f"\n[Main] Cross Validation with {self.n_cv_splits} folds "
              f"(approx each val size: {approx_val_size})")
        print(f"(calculate_competition_score 사용, 기본 k=15)")

        y_all = train_Y_series.values

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")

            X_train_fold = X_train_final[train_idx]
            y_train_fold = y_all[train_idx]

            X_val = X_train_final[val_idx]
            y_val = y_all[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            model = MainModel(n_estimators=200, random_state=42 + fold_idx, n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)

            val_prob_ng = model.predict_proba(X_val)[:, 1]

            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
            )

            print(f"[Fold {fold_idx+1}] ROC={roc:.6f} Profit={profit:.2f} Score={score:.6f}")
            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (StratifiedKFold, Encoder+AE+RF) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 12. 제출용 모델 재학습 (Train 전체 사용)
        self.main_model = MainModel(n_estimators=200, random_state=42, n_jobs=-1)
        self.main_model.fit(X_train_final, y_all)

        # 13. Test 예측
        test_prob = self.main_model.predict_proba(X_test_final)[:, 1]
        print(f"\n[Main] Test prob range: {test_prob.min():.4f} ~ {test_prob.max():.4f}")

        # 14. 제출 파일 생성
        submission = pd.read_csv("../data/submission/sample_submission.csv")
        submission['probability'] = np.concatenate([test_prob, test_prob])
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/CNN_AE_RF_submission_{timestamp}.csv"

        submission.to_csv(save_path, index=False)
        print(f"[Main] Saved submission to {save_path}")

        selected_count = submission['decision'].sum()
        print(f"[Main] Total selected products: {selected_count}")

        logger.close()
        sys.stdout = sys.__stdout__
        print(f"[Main] Log saved to: {logger.log_path}")

        return submission


def main():
    pipeline = ProductionPipeline(
        n_epochs=13,
        batch_size=32,
        n_cv_splits=5,
        encoder_weight_path="../weight/feature_encoder.pth"
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()