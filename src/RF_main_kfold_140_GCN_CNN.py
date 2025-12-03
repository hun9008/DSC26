# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# CNN encoder 쪽
from CNN_encoder import (
    DataProcessor,
    SpatialRasterizer,
    FeatureEncoder as CNNFeatureEncoder,
    MultiModalDataset as CNNMultiModalDataset,
)

# GCN encoder 쪽
from GCN_encoder import (
    FeatureEncoder as GCNFeatureEncoder,
    MultiModalDataset as GCNMultiModalDataset,
)

# 평가 함수 공통 모듈
from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

from util.logger import TeeLogger


# ----------------------------------------------------
# 5. Main Model (RandomForest 기반 이진 분류기)
# ----------------------------------------------------
class MainModel:
    """
    Main Model:
      - 입력: (CNN+GCN에서 추출한) feature
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
# 6. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """(사전 학습된) CNN FeatureEncoder + GCN FeatureEncoder + MainModel 하이브리드 파이프라인"""

    def __init__(
        self,
        n_epochs=13,
        batch_size=32,
        n_cv_splits=5,
        encoder_weight_path="../weight/feature_encoder.pth",
        gcn_encoder_weight_path="../weight/GCN_feature_encoder_20251129_215146.pth",
        cnn_weight=0.5,
        gcn_weight=0.5,
    ):
        self.data_processor = DataProcessor()
        self.rasterizer = None

        self.cnn_encoder = None
        self.gcn_encoder = None
        self.main_model = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits

        self.encoder_weight_path = encoder_weight_path
        self.gcn_encoder_weight_path = gcn_encoder_weight_path

        # CNN / GCN feature 가중치
        total = cnn_weight + gcn_weight
        if total <= 0:
            raise ValueError("cnn_weight + gcn_weight must be > 0")
        self.cnn_weight = cnn_weight / total
        self.gcn_weight = gcn_weight / total

    # --------- Encoder 로드 ---------
    def load_pretrained_encoders(self, basic_feature_dim):
        # CNN encoder
        if not os.path.exists(self.encoder_weight_path):
            raise FileNotFoundError(
                f"CNN encoder weight not found: {self.encoder_weight_path}\n"
                f"먼저 CNN encoder_train.py 를 실행해서 feature_encoder.pth 를 생성하세요."
            )

        self.cnn_encoder = CNNFeatureEncoder(
            basic_feature_dim=basic_feature_dim
        ).to(self.device)

        state_dict_cnn = torch.load(self.encoder_weight_path, map_location=self.device)
        self.cnn_encoder.load_state_dict(state_dict_cnn)
        self.cnn_encoder.eval()
        print(f"[Main] Loaded CNN encoder weights from {self.encoder_weight_path}")

        # GCN encoder
        if not os.path.exists(self.gcn_encoder_weight_path):
            raise FileNotFoundError(
                f"GCN encoder weight not found: {self.gcn_encoder_weight_path}\n"
                f"먼저 GCN encoder_train.py 를 실행해서 해당 weight 를 생성하세요."
            )

        self.gcn_encoder = GCNFeatureEncoder(
            basic_feature_dim=basic_feature_dim
        ).to(self.device)

        state_dict_gcn = torch.load(self.gcn_encoder_weight_path, map_location=self.device)
        self.gcn_encoder.load_state_dict(state_dict_gcn)
        self.gcn_encoder.eval()
        print(f"[Main] Loaded GCN encoder weights from {self.gcn_encoder_weight_path}")

    # --------- 공통 feature 추출 함수 ---------
    @staticmethod
    def extract_features(loader, encoder, device, is_test=False):
        """
        loader 가 무엇을 내보내든 (img,basic or pts,basic),
        encoder.extract_features(modal, basic) 만 동작하면 사용 가능.
        """
        encoder.eval()
        all_features = []
        with torch.no_grad():
            if is_test:
                for modal, basic in loader:
                    modal = modal.to(device)
                    basic = basic.to(device)
                    feats = encoder.extract_features(modal, basic)
                    all_features.append(feats.cpu().numpy())
            else:
                for modal, basic, _ in loader:
                    modal = modal.to(device)
                    basic = basic.to(device)
                    feats = encoder.extract_features(modal, basic)
                    all_features.append(feats.cpu().numpy())
        return np.concatenate(all_features, axis=0)

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

        print("[Main] Start Production Pipeline")
        print(f"[Main] CNN/GCN weights: cnn={self.cnn_weight:.3f}, gcn={self.gcn_weight:.3f}")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정 (CNN에서 사용, GCN은 인터페이스만 맞추기용)
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)

        # 4. 기본 피처 전처리
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"[Main] Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"[Main] Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. Dataset / DataLoader (feature 추출용)
        # CNN용
        cnn_train_dataset = CNNMultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        cnn_test_dataset = CNNMultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )

        # GCN용 (rasterizer는 사용하지 않지만 인터페이스 유지)
        gcn_train_dataset = GCNMultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        gcn_test_dataset = GCNMultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )

        cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=self.batch_size, shuffle=False)
        cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=self.batch_size, shuffle=False)

        gcn_train_loader = DataLoader(gcn_train_dataset, batch_size=self.batch_size, shuffle=False)
        gcn_test_loader = DataLoader(gcn_test_dataset, batch_size=self.batch_size, shuffle=False)

        # 6. 사전 학습된 CNN / GCN FeatureEncoder 로드
        self.load_pretrained_encoders(self.data_processor.basic_feature_dim)

        # 7. FeatureEncoder를 이용해 feature 추출
        print("[Main] Extract CNN features...")
        X_train_feat_cnn = self.extract_features(
            cnn_train_loader, self.cnn_encoder, self.device, is_test=False
        )
        X_test_feat_cnn = self.extract_features(
            cnn_test_loader, self.cnn_encoder, self.device, is_test=True
        )

        print("[Main] Extract GCN features...")
        X_train_feat_gcn = self.extract_features(
            gcn_train_loader, self.gcn_encoder, self.device, is_test=False
        )
        X_test_feat_gcn = self.extract_features(
            gcn_test_loader, self.gcn_encoder, self.device, is_test=True
        )

        print(f"[Main] CNN features (Train): {X_train_feat_cnn.shape}")
        print(f"[Main] GCN features (Train): {X_train_feat_gcn.shape}")
        print(f"[Main] CNN features (Test) : {X_test_feat_cnn.shape}")
        print(f"[Main] GCN features (Test) : {X_test_feat_gcn.shape}")

        # 8. CNN / GCN weighted sum 으로 하이브리드 피처 생성
        X_train_hybrid = (
            self.cnn_weight * X_train_feat_cnn +
            self.gcn_weight * X_train_feat_gcn
        )
        X_test_hybrid = (
            self.cnn_weight * X_test_feat_cnn +
            self.gcn_weight * X_test_feat_gcn
        )

        print(f"[Main] Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"[Main] Hybrid features (Test) : {X_test_hybrid.shape}")

        # 9. Cross Validation (Stratified K-Fold)
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

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")

            X_train_fold = X_train_hybrid[train_idx]
            y_train_fold = train_Y_series.values[train_idx]

            X_val = X_train_hybrid[val_idx]
            y_val = train_Y_series.values[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            model = MainModel(n_estimators=200, random_state=42 + fold_idx, n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)

            val_prob_ng = model.predict_proba(X_val)[:, 1]

            # 공통 평가 함수 사용 (k는 기본값 15 사용)
            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
            )

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (StratifiedKFold, approx val~146) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 10. 제출용 모델 재학습 (Train 전체 사용)
        self.main_model = MainModel(n_estimators=200, random_state=42, n_jobs=-1)
        self.main_model.fit(X_train_hybrid, train_Y_series.values)

        # 11. Test 예측
        test_prob = self.main_model.predict_proba(X_test_hybrid)[:, 1]
        print(f"\n[Main] Test prob range: {test_prob.min():.4f} ~ {test_prob.max():.4f}")

        # 12. 제출 파일 생성
        submission = pd.read_csv("../data/submission/sample_submission.csv")
        submission['probability'] = np.concatenate([test_prob, test_prob])
        submission['decision'] = False

        n_sub = len(submission)
        half_sub = n_sub // 2

        idx_L_sub = submission.index[:half_sub]
        idx_P_sub = submission.index[half_sub:]

        # 여기서는 기존 로직 유지 (L/P 각각 170개 선택)
        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/CNN_GCN_RF_submission_{timestamp}.csv"

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
        encoder_weight_path="../weight/feature_encoder.pth",  # CNN encoder
        gcn_encoder_weight_path="../weight/GCN_feature_encoder_20251129_215146.pth",  # GCN encoder
        cnn_weight=0.5,
        gcn_weight=0.5,
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()