# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

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
    """개선된 ImageCNN: GAP 적용 + 파라미터 최적화 (RF_main_cnn_extractor.py와 동일 구조)"""
    def __init__(self, output_dim=64, input_size=32):
        super(ImageCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
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
    RF_main_cnn_extractor에서 사용한 FullE2EModel 구조
    - image_cnn: rasterized image -> 64-dim
    - basic_mlp: tabular -> 64-dim
    - head: 128 -> 1
    여기서는 학습된 weight를 로드해서 `image_cnn`만 feature extractor로 사용.
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
# 5. Main Model (MLP 기반 이진 분류기)
# ----------------------------------------------------
class MainModel(nn.Module):
    """
    Main Model:
      - 입력: [기본 피처, CNN 임베딩] 하이브리드 feature
      - 모델: MLP
      - 출력: NG 확률 (sigmoid(logit))
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_feat=False):
        """
        x: (B, D)
        return_feat=True면 hidden feature도 함께 반환 (KL 대상)
        """
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        logit = self.fc2(h)  # (B, 1)
        if return_feat:
            return logit, h
        return logit


# ----------------------------------------------------
# 6. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """
    (사전 학습된) FullE2EModel 의 image_cnn + MainModel(MLP) 하이브리드 파이프라인

    - x,y,p → SpatialRasterizer(grid_size=32, cubic) → 1x32x32 image
    - image → 사전학습 CNN(image_cnn, weight=best_model_3232.pth) → 64-dim
    - 기본 tabular feature → DataProcessor.preprocess_basic → D_basic-dim
    - [basic, cnn_feat] concat → MLP(+KL) 메인 모델
    """

    def __init__(self, n_epochs=30, batch_size=64, n_cv_splits=5,
                 encoder_weight_path="../weight/best_model_3232.pth",
                 hidden_dim=256, dropout=0.2,
                 lr=1e-3, lambda_align=1e-3):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.cnn_model = None  # FullE2EModel (여기서 image_cnn만 사용)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.lambda_align = lambda_align  # KL loss 가중치

    # ---------------- 사전 학습 CNN Extractor 로드 ----------------
    def load_pretrained_cnn_extractor(self, basic_feature_dim, input_grid_size=32):
        if not os.path.exists(self.encoder_weight_path):
            raise FileNotFoundError(
                f"Pretrained CNN weight not found: {self.encoder_weight_path}\n"
                f"먼저 RF_main_cnn_extractor.py 를 실행해서 best_model_3232.pth 를 생성하세요."
            )

        self.cnn_model = FullE2EModel(
            basic_feature_dim=basic_feature_dim,
            image_cnn_output_dim=64,
            basic_mlp_output_dim=64,
            input_grid_size=input_grid_size,
        ).to(self.device)

        state_dict = torch.load(self.encoder_weight_path, map_location=self.device)
        self.cnn_model.load_state_dict(state_dict)
        self.cnn_model.eval()
        print(f"[Main] Loaded CNN extractor weights from {self.encoder_weight_path}")

    # ---------------- CNN feature 추출 ----------------
    def extract_cnn_features(self, loader, is_test=False):
        """
        loader: MultiModalDataset 기반 DataLoader
          - train: (img, basic, label)
          - test : (img, basic)
        출력: numpy array, shape (N, 64)
        """
        assert self.cnn_model is not None, "cnn_model이 로드되지 않았습니다."
        self.cnn_model.eval()

        all_features = []
        with torch.no_grad():
            if is_test:
                for img, basic in loader:
                    img = img.to(self.device)
                    # basic은 CNN 추출에는 사용하지 않음
                    img_feat = self.cnn_model.image_cnn(img)  # (B, 64)
                    all_features.append(img_feat.cpu().numpy())
            else:
                for img, basic, _ in loader:
                    img = img.to(self.device)
                    img_feat = self.cnn_model.image_cnn(img)  # (B, 64)
                    all_features.append(img_feat.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    # ---------------- Stratified K-Fold CV splits 생성 ----------------
    @staticmethod
    def make_cv_splits(y_series, n_splits=5, base_seed=42):
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

    # ---------------- KL(Gaussian) loss 계산 ----------------
    @staticmethod
    def gaussian_kl(feat_s, feat_t, eps=1e-6):
        """
        feat_s, feat_t: (B, D)
        배치 기준 mean/var로 가우시안 근사 후 KL(P_s || P_t)
        """
        mu_s = feat_s.mean(dim=0)
        var_s = feat_s.var(dim=0, unbiased=False) + eps

        mu_t = feat_t.mean(dim=0)
        var_t = feat_t.var(dim=0, unbiased=False) + eps

        kl = 0.5 * torch.sum(
            torch.log(var_t / var_s)
            + (var_s + (mu_s - mu_t) ** 2) / var_t
            - 1.0
        )
        return kl

    # ---------------- MLP + KL 학습 함수 ----------------
    def train_mlp_with_kl(self, X_train, y_train, X_target, verbose_prefix=""):
        """
        X_train: (N_s, D) numpy  (source: train hybrid feature)
        y_train: (N_s,) numpy
        X_target: (N_t, D) numpy  (target: test hybrid feature, label 없음)
        """
        input_dim = X_train.shape[1]

        model = MainModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()

        # TensorDataset
        Xs = torch.from_numpy(X_train).float()
        ys = torch.from_numpy(y_train).float()
        Xt = torch.from_numpy(X_target).float()

        train_dataset = TensorDataset(Xs, ys)
        target_dataset = TensorDataset(Xt)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        target_iter = iter(target_loader)

        for epoch in range(1, self.n_epochs + 1):
            model.train()
            epoch_loss = epoch_cls = epoch_kl = 0.0
            n_steps = 0

            for xs_batch, ys_batch in train_loader:
                try:
                    (xt_batch,) = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    (xt_batch,) = next(target_iter)

                xs_batch = xs_batch.to(self.device)
                ys_batch = ys_batch.to(self.device)
                xt_batch = xt_batch.to(self.device)

                # forward
                logit_s, feat_s = model(xs_batch, return_feat=True)
                logit_t, feat_t = model(xt_batch, return_feat=True)

                logit_s = logit_s.squeeze(-1)  # (B,)

                # 분류 loss (source = train)
                L_cls = bce(logit_s, ys_batch)

                # feature 분포 KL(source vs target = test)
                L_kl = self.gaussian_kl(feat_s, feat_t)

                # total loss
                L_total = L_cls + self.lambda_align * L_kl

                optimizer.zero_grad()
                L_total.backward()
                optimizer.step()

                epoch_loss += L_total.item()
                epoch_cls += L_cls.item()
                epoch_kl += L_kl.item()
                n_steps += 1

            if n_steps > 0:
                epoch_loss /= n_steps
                epoch_cls /= n_steps
                epoch_kl /= n_steps

            print(
                f"{verbose_prefix}Epoch [{epoch}/{self.n_epochs}] "
                f"Loss={epoch_loss:.4f}  CLS={epoch_cls:.4f}  KL={epoch_kl:.4f}"
            )

        return model

    # ---------------- MLP 예측 함수 ----------------
    def predict_proba_with_model(self, model, X):
        model.eval()
        X_tensor = torch.from_numpy(X).float()
        probs = []
        with torch.no_grad():
            loader = DataLoader(X_tensor, batch_size=self.batch_size, shuffle=False)
            for xb in loader:
                xb = xb.to(self.device)
                logit = model(xb)  # (B, 1)
                logit = logit.squeeze(-1)
                p = torch.sigmoid(logit)
                probs.append(p.cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        return probs

    # ---------------- 전체 파이프라인 실행 ----------------
    def run_production_pipeline(self):

        logger = TeeLogger()
        sys.stdout = logger

        print("[Main] Start Production Pipeline (CNN Extractor + MLP + KL)")
        print("[Main] x,y,p → CNN(64-d) + basic feature concat → MLP+KL")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정 (32x32, cubic 보간)
        self.rasterizer = SpatialRasterizer(
            x_min, x_max, y_min, y_max,
            grid_size=32,
            interpolation_method='cubic'
        )

        # 4. 기본 피처 전처리
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"[Main] Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"[Main] Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. Dataset / DataLoader (CNN feature 추출용)
        train_dataset = MultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        test_dataset = MultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )

        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 6. 사전 학습된 CNN Extractor 로드
        self.load_pretrained_cnn_extractor(
            basic_feature_dim=self.data_processor.basic_feature_dim,
            input_grid_size=32
        )

        # 7. CNN을 이용해 x,y,p → 64-d feature 추출
        X_train_cnn_feat = self.extract_cnn_features(train_loader_seq, is_test=False)
        X_test_cnn_feat = self.extract_cnn_features(test_loader, is_test=True)

        print(f"[Main] CNN features (Train): {X_train_cnn_feat.shape}")
        print(f"[Main] CNN features (Test) : {X_test_cnn_feat.shape}")

        # 8. 하이브리드 피처 생성 (기본 피처 + CNN 64차원)
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_cnn_feat], axis=1)
        X_test_hybrid = np.concatenate([X_test_basic_np, X_test_cnn_feat], axis=1)

        print(f"[Main] Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"[Main] Hybrid features (Test) : {X_test_hybrid.shape}")
        print(f"  - 기본 피처 차원: {X_train_basic_np.shape[1]}")
        print(f"  - CNN 피처 차원: {X_train_cnn_feat.shape[1]}")

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

            # fold별 MLP + KL 학습 (target 도메인은 항상 전체 test hybrid feature)
            model_fold = self.train_mlp_with_kl(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_target=X_test_hybrid,
                verbose_prefix=f"[Fold {fold_idx+1}] "
            )

            # validation 예측
            val_prob_ng = self.predict_proba_with_model(model_fold, X_val)

            # 공통 평가 함수 사용
            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
            )

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (StratifiedKFold) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 10. 제출용 모델 재학습 (Train 전체 + KL 정렬)
        print("\n[Main] Train final MLP with KL (full train, hybrid feature)")
        final_model = self.train_mlp_with_kl(
            X_train=X_train_hybrid,
            y_train=train_Y_series.values,
            X_target=X_test_hybrid,
            verbose_prefix="[Final] "
        )

        # 11. Test 예측
        test_prob = self.predict_proba_with_model(final_model, X_test_hybrid)
        print(f"\n[Main] Test prob range: {test_prob.min():.4f} ~ {test_prob.max():.4f}")

        # 12. 제출 파일 생성
        submission = pd.read_csv("../data/submission/sample_submission.csv")
        submission['probability'] = np.concatenate([test_prob, test_prob])
        submission['decision'] = False

        n_sub = len(submission)
        half_sub = n_sub // 2

        idx_L_sub = submission.index[:half_sub]
        idx_P_sub = submission.index[half_sub:]

        # 기존 로직 유지 (L/P 각각 170개 선택, NG 확률 낮은 것부터 선택)
        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/CNN_MLP_KL_shallow_submission_{timestamp}.csv"

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
        n_epochs=30,
        batch_size=64,
        n_cv_splits=5,
        encoder_weight_path="../weight/best_model_3232.pth",
        hidden_dim=256,
        dropout=0.2,
        lr=1e-3,
        lambda_align=1e-3,
    )
    submission_result = pipeline.run_production_pipeline()

    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()