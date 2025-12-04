# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from tabpfn import TabPFNClassifier  # TabPFN

# ➜ RF_main_cnn_extractor에서 쓰던 cubic rasterizer 버전 사용
from CNN_cubic_encoder import (
    DataProcessor,
    SpatialRasterizer,
    MultiModalDataset,
)

# 평가 함수 공통 모듈
from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

from util.logger import TeeLogger


# ----------------------------------------------------
# 1. CNN Encoder 구조 (RF_main_cnn_extractor.py와 동일)
# ----------------------------------------------------
class ImageCNN(nn.Module):
    """개선된 ImageCNN: GAP 적용 + 파라미터 최적화"""
    def __init__(self, output_dim=64, input_size=32):
        super(ImageCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 4x4 -> 2x2 (채널 128 유지)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # GAP 이후 처리
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),  # 128 -> 64
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.features(x)   # (B, 128, H', W')
        x = self.gap(x)        # (B, 128, 1, 1)
        return self.fc_out(x)  # (B, 64)


class FullE2EModel(nn.Module):
    """
    RF_main_cnn_extractor에서 사용한 FullE2EModel 구조 그대로
    - image_cnn: rasterized image -> 64-dim
    - basic_mlp: tabular -> 64-dim
    - head: 128 -> 1 (학습 시 사용, 여기서는 feature만 사용)
    """
    def __init__(self, basic_feature_dim,
                 image_cnn_output_dim=64,
                 basic_mlp_output_dim=64,
                 input_grid_size=32):
        super(FullE2EModel, self).__init__()

        # 1. Image Branch
        self.image_cnn = ImageCNN(output_dim=image_cnn_output_dim,
                                  input_size=input_grid_size)

        # 2. Tabular Branch
        self.basic_mlp = nn.Sequential(
            nn.Linear(basic_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, basic_mlp_output_dim),
            nn.ReLU(),
        )

        # 3. Fusion Head
        combined_dim = image_cnn_output_dim + basic_mlp_output_dim  # 64 + 64 = 128

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)      # (B, 64)
        basic_feat = self.basic_mlp(x_basic)    # (B, 64)
        combined = torch.cat((img_feat, basic_feat), dim=1)  # (B, 128)
        output = self.head(combined)
        return output


# ----------------------------------------------------
# 2. Main Model (TabPFN 기반 이진 분류기)
# ----------------------------------------------------
class MainModel:
    """
    Main Model: TabPFNClassifier 기반
    """

    def __init__(self, device=None, model_path=None, **kwargs):
        """
        device: "cpu" 또는 "cuda"
        model_path: 오픈 버전 TabPFN ckpt 경로
                    예: "../weight/tabpfn_open.ckpt"
        kwargs: 호출부에서 넘기는 random_state, N_ensemble_configurations 등 무시
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path is not None:
            self.model = TabPFNClassifier(
                device=device,
                model_path=model_path,
            )
        else:
            self.model = TabPFNClassifier(
                device=device,
            )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model.fit(X, y)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 3. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """(사전 학습된) FullE2EModel + MainModel(TabPFN) 하이브리드 파이프라인"""

    def __init__(self, n_epochs=13, batch_size=32, n_cv_splits=5,
                 encoder_weight_path="../weight/best_model_3232.pth",
                 tabpfn_model_path=None):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None  # FullE2EModel 인스턴스
        self.main_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path
        # 오픈 TabPFN ckpt 경로
        self.tabpfn_model_path = tabpfn_model_path

    # ---------------- 사전 학습된 encoder 로드 ----------------
    def load_pretrained_encoder(self, basic_feature_dim):
        if not os.path.exists(self.encoder_weight_path):
            raise FileNotFoundError(
                f"Pretrained encoder weight not found: {self.encoder_weight_path}\n"
                f"RF_main_cnn_extractor.py 를 실행해서 best_model_3232.pth 를 생성하세요."
            )

        # RF_main_cnn_extractor와 동일한 FullE2EModel 구조로 생성
        self.feature_encoder = FullE2EModel(
            basic_feature_dim=basic_feature_dim,
            image_cnn_output_dim=64,
            basic_mlp_output_dim=64,
            input_grid_size=32,
        ).to(self.device)

        state_dict = torch.load(self.encoder_weight_path, map_location=self.device)
        self.feature_encoder.load_state_dict(state_dict)
        self.feature_encoder.eval()
        print(f"[Main] Loaded encoder weights from {self.encoder_weight_path}")

    # ---------------- feature 추출 (image + basic MLP concat) ----------------
    def extract_features(self, loader, is_test=False):
        assert self.feature_encoder is not None, "feature_encoder가 로드되지 않았습니다."
        self.feature_encoder.eval()

        all_features = []
        with torch.no_grad():
            if is_test:
                for img, basic in loader:
                    img = img.to(self.device)
                    basic = basic.to(self.device)
                    img_feat = self.feature_encoder.image_cnn(img)      # (B, 64)
                    basic_feat = self.feature_encoder.basic_mlp(basic)  # (B, 64)
                    feats = torch.cat((img_feat, basic_feat), dim=1)    # (B, 128)
                    all_features.append(feats.cpu().numpy())
            else:
                for img, basic, _ in loader:
                    img = img.to(self.device)
                    basic = basic.to(self.device)
                    img_feat = self.feature_encoder.image_cnn(img)
                    basic_feat = self.feature_encoder.basic_mlp(basic)
                    feats = torch.cat((img_feat, basic_feat), dim=1)
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

        print("[Main] Start Production Pipeline (TabPFN + cubic CNN encoder)")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정 (Cubic 보간, 32x32 그리드 - RF_main_cnn_extractor와 동일)
        self.rasterizer = SpatialRasterizer(
            x_min, x_max, y_min, y_max,
            grid_size=32,
            interpolation_method='cubic',
        )

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

        # 6. 사전 학습된 FullE2EModel 로드
        self.load_pretrained_encoder(self.data_processor.basic_feature_dim)

        # 7. encoder를 이용해 feature 추출 (image CNN + basic MLP -> 128차원)
        X_train_feat = self.extract_features(train_loader_seq, is_test=False)
        X_test_feat = self.extract_features(test_loader, is_test=True)

        print(f"[Main] Encoded features (Train): {X_train_feat.shape}")
        print(f"[Main] Encoded features (Test) : {X_test_feat.shape}")

        # 8. TabPFN 입력 피처 설정 (128차원)
        X_train_hybrid = X_train_feat.astype(np.float32)
        X_test_hybrid = X_test_feat.astype(np.float32)

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

            # TabPFN 기반 MainModel 사용
            model = MainModel(
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_path=self.tabpfn_model_path,
                random_state=42 + fold_idx,  # 무시되지만 인터페이스 유지
            )
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
        self.main_model = MainModel(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_path=self.tabpfn_model_path,
            random_state=42,
        )
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
        save_path = f"../data/submission/TabPFN_lightCNN_{timestamp}.csv"

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
        encoder_weight_path="../weight/best_model_3232.pth",
        # 오픈 TabPFN ckpt 경로 지정 시 사용
        # 예시: "../weight/tabpfn_open.ckpt"
        tabpfn_model_path=None,
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()