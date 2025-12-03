# RF_main_cnn_extractor.py
# 제공된 코드의 CNN Extractor 구조를 사용한 하이브리드 모델
# ⚠️ 중요: OpenMP 충돌 해결 - 모든 import 전에 설정해야 함
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = "1"


import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
# 1. CNN Extractor (제공된 코드 구조 사용)
# ----------------------------------------------------
class ImageCNN(nn.Module):
    """개선된 ImageCNN: GAP 적용 + 파라미터 최적화"""
    def __init__(self, output_dim=64, input_size=32): 
        super(ImageCNN, self).__init__()
        
        # [구조 개선 1] 채널 수 최적화 (Overfitting 방지)
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: 8x8 -> 4x4 (채널 128 유지)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # [구조 개선 2] GAP (Global Average Pooling) 도입
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # GAP 이후 처리 (Dropout 강화)
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim), # 128 -> 64
            nn.ReLU(),
            nn.Dropout(0.4)
        )
    
    def forward(self, x):
        x = self.features(x)   # (Batch, 128, 4, 4)
        x = self.gap(x)        # (Batch, 128, 1, 1)
        return self.fc_out(x)  # (Batch, 64)


class FullE2EModel(nn.Module):
    """개선된 FullE2EModel: MLP 균형 조절 + GAP 적용"""
    def __init__(self, basic_feature_dim, image_cnn_output_dim=64, basic_mlp_output_dim=64, input_grid_size=32):
        super(FullE2EModel, self).__init__()
        
        # 1. Image Branch (개선된 CNN)
        self.image_cnn = ImageCNN(output_dim=image_cnn_output_dim, input_size=input_grid_size)
        
        # [구조 개선 3] Tabular Branch (MLP) 용량 확대
        self.basic_mlp = nn.Sequential(
            nn.Linear(basic_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, basic_mlp_output_dim),
            nn.ReLU()
        )
        
        # 3. Fusion Head
        combined_dim = image_cnn_output_dim + basic_mlp_output_dim
        
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)   # (Batch, 64)
        basic_feat = self.basic_mlp(x_basic) # (Batch, 64)
        combined = torch.cat((img_feat, basic_feat), dim=1) # (Batch, 128)
        output = self.head(combined)
        return output


# ----------------------------------------------------
# 2. Main Model (RandomForest 기반 이진 분류기)
# ----------------------------------------------------
class MainModel:
    """
    Main Model:
      - 입력: 기본 피처 + Image CNN 피처 (64차원)
      - 모델: RandomForestClassifier (기존 하이퍼파라미터 유지)
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
# 3. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """CNN Extractor (Image CNN만 추출) + MainModel 하이브리드 파이프라인"""

    def __init__(self, n_epochs=20, batch_size=32, n_cv_splits=5,
                 encoder_weight_path="best_model.pth"):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.cnn_model = None  # FullE2EModel
        self.main_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path

    def train_cnn_extractor(self, train_loader, val_loader=None):
        """E2E 모델을 '학습'시켜 피처 추출기(best_model.pth)를 만듭니다. (Validation set으로 최적 epoch 결정)"""
        
        self.cnn_model = FullE2EModel(
            self.data_processor.basic_feature_dim, 
            image_cnn_output_dim=64,
            basic_mlp_output_dim=64,
            input_grid_size=32
        ).to(self.device)
        
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss() 
        
        if val_loader is not None:
            print("\n[Main] CNN 피처 추출기 학습 시작 (Validation set으로 최적 epoch 결정)...")
            best_val_loss = float('inf')
            best_epoch = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.n_epochs):
                # Train
                self.cnn_model.train()
                train_loss_total = 0.0
                
                for img, basic, labels in train_loader:
                    img, basic, labels = img.to(self.device), basic.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.cnn_model(img, basic)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss_total += loss.item()
                
                avg_train_loss = train_loss_total / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation
                self.cnn_model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for img, basic, labels in val_loader:
                        img, basic, labels = img.to(self.device), basic.to(self.device), labels.to(self.device)
                        outputs = self.cnn_model(img, basic)
                        loss = criterion(outputs, labels)
                        val_loss_total += loss.item()
                
                avg_val_loss = val_loss_total / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # 최적 모델 저장
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(self.cnn_model.state_dict(), self.encoder_weight_path)
                
                print(f"  Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} "
                      f"{'⭐ Best' if epoch + 1 == best_epoch else ''}")
            
            print(f"\n✅ CNN 피처 추출기 학습 완료.")
            print(f"   - 최적 Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
            print(f"   - 가중치 저장: {self.encoder_weight_path}")
            print(f"   - Train Loss 변화: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
            print(f"   - Val Loss 변화: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
            
            # 최적 모델 로드
            self.cnn_model.load_state_dict(torch.load(self.encoder_weight_path, map_location=self.device))
        else:
            print("\n[Main] CNN 피처 추출기 학습 시작 (Train 전체 사용, Validation 없음)...")
            
            for epoch in range(self.n_epochs):
                self.cnn_model.train()
                train_loss_total = 0.0
                
                for img, basic, labels in train_loader:
                    img, basic, labels = img.to(self.device), basic.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.cnn_model(img, basic)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss_total += loss.item()
                
                avg_train_loss = train_loss_total / len(train_loader)
                print(f"  Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {avg_train_loss:.4f}")
            
            # 마지막 epoch 모델 저장
            torch.save(self.cnn_model.state_dict(), self.encoder_weight_path)
            print(f"✅ CNN 피처 추출기 학습 완료. (가중치 저장: {self.encoder_weight_path})")

    def extract_cnn_features(self, loader, is_test=False):
        """'학습된' E2E 모델을 사용해 'Image CNN 피처'만 추출합니다 (64차원)."""
        
        # 저장된 최고 성능 모델 불러오기
        if self.cnn_model is None:
            self.cnn_model = FullE2EModel(
                self.data_processor.basic_feature_dim,
                image_cnn_output_dim=64,
                basic_mlp_output_dim=64,
                input_grid_size=32
            ).to(self.device)
            self.cnn_model.load_state_dict(torch.load(self.encoder_weight_path, map_location=self.device))
        
        self.cnn_model.eval()
        
        all_features = []
        with torch.no_grad():
            if is_test:
                for img, basic in loader:
                    img, basic = img.to(self.device), basic.to(self.device)
                    # Image CNN만 추출 (64차원, MLP 제외)
                    img_feat = self.cnn_model.image_cnn(img)
                    all_features.append(img_feat.cpu().numpy())
            else:
                for img, basic, labels in loader:  # 레이블은 사용 안함
                    img, basic = img.to(self.device), basic.to(self.device)
                    # Image CNN만 추출 (64차원, MLP 제외)
                    img_feat = self.cnn_model.image_cnn(img)
                    all_features.append(img_feat.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)

    # ---------------- Validation 인덱스 생성 (NG 15, Good 45) ----------------
    @staticmethod
    def make_fixed_validation_indices(y_series, n_ng_val=15, n_good_val=45, seed=42):
        rng = np.random.RandomState(seed)
        y = y_series.values
        all_idx = np.arange(len(y))

        ng_idx = all_idx[y == 1]
        good_idx = all_idx[y == 0]

        if len(ng_idx) < n_ng_val or len(good_idx) < n_good_val:
            raise ValueError("Validation에 필요한 NG 또는 Good 샘플 수가 부족합니다.")

        rng.shuffle(ng_idx)
        rng.shuffle(good_idx)

        val_ng_idx = ng_idx[:n_ng_val]
        val_good_idx = good_idx[:n_good_val]

        val_idx = np.concatenate([val_ng_idx, val_good_idx])
        rng.shuffle(val_idx)

        train_idx = np.setdiff1d(all_idx, val_idx)

        return train_idx, val_idx

    def make_cv_splits(self, y_series, n_splits=5, n_ng_val=15, n_good_val=45, base_seed=42):
        splits = []
        for fold in range(n_splits):
            seed = base_seed + fold
            train_idx, val_idx = self.make_fixed_validation_indices(
                y_series,
                n_ng_val=n_ng_val,
                n_good_val=n_good_val,
                seed=seed
            )
            splits.append((train_idx, val_idx))
        return splits

    # ---------------- 전체 파이프라인 실행 ----------------
    def run_production_pipeline(self):

        logger = TeeLogger()
        sys.stdout = logger

        print("[Main] Start Production Pipeline (CNN Extractor Version)")
        print("[Main] CNN Extractor: Image CNN만 추출 (64차원, MLP 제외)")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정 (Cubic 보간법 사용, 32x32 그리드)
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=32, interpolation_method='cubic')

        # 4. 기본 피처 전처리 (Train 전체로 fit)
        print("\n[Main] 기본 피처 전처리 (Train 전체로 fit)")
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"[Main] Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"[Main] Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. CNN 학습용 Dataset / DataLoader 생성 (Train 전체)
        print("\n[Main] CNN 학습용 Dataset / DataLoader 생성 (Train 전체)")
        train_dataset_cnn = MultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=self.batch_size, shuffle=True)
        
        print(f"  - Train size: {len(train_df)} (NG={(train_Y_series.values==1).sum()}, Good={(train_Y_series.values==0).sum()})")

        # 6. CNN 피처 추출기 학습 (Train 전체 사용)
        if not os.path.exists(self.encoder_weight_path):
            print(f"\n[Main] CNN Extractor 학습 시작 (Train 전체 사용, 가중치 파일 없음)")
            self.train_cnn_extractor(train_loader_cnn, val_loader=None)
        else:
            print(f"\n[Main] 기존 CNN Extractor 가중치 로드: {self.encoder_weight_path}")
            self.cnn_model = FullE2EModel(
                self.data_processor.basic_feature_dim,
                image_cnn_output_dim=64,
                basic_mlp_output_dim=64,
                input_grid_size=32
            ).to(self.device)
            self.cnn_model.load_state_dict(torch.load(self.encoder_weight_path, map_location=self.device))

        # 7. CNN 피처 추출 (전체 Train 데이터 및 Test 데이터)
        print("\n[Main] CNN 피처 추출 (전체 Train 데이터 및 Test 데이터)")
        # 전체 train 데이터셋 생성 (CNN 학습용 split과 별개)
        train_dataset_full = MultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        train_loader_full = DataLoader(train_dataset_full, batch_size=self.batch_size, shuffle=False)
        
        test_dataset = MultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        X_train_cnn_feat = self.extract_cnn_features(train_loader_full, is_test=False)
        X_test_cnn_feat = self.extract_cnn_features(test_loader, is_test=True)

        print(f"  추출된 CNN 피처 형태 (Train): {X_train_cnn_feat.shape}")
        print(f"  추출된 CNN 피처 형태 (Test) : {X_test_cnn_feat.shape}")

        # 8. 하이브리드 피처 생성 (원본 기본 피처 + Image CNN 64차원)
        print("\n[Main] 하이브리드 피처 결합 (원본 기본 피처 + Image CNN 64차원)")
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_cnn_feat], axis=1)
        X_test_hybrid = np.concatenate([X_test_basic_np, X_test_cnn_feat], axis=1)

        print(f"  하이브리드 피처 형태 (Train): {X_train_hybrid.shape}")
        print(f"    - 원본 기본 피처: {X_train_basic_np.shape[1]}차원")
        print(f"    - Image CNN 피처: {X_train_cnn_feat.shape[1]}차원 (MLP 제외)")
        print(f"  하이브리드 피처 형태 (Test) : {X_test_hybrid.shape}")

        # 9. Cross Validation (전체 데이터 사용)
        cv_splits = self.make_cv_splits(
            train_Y_series,
            n_splits=self.n_cv_splits,
            n_ng_val=15,
            n_good_val=45,
            base_seed=42
        )

        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []

        print(f"\n[Main] Cross Validation with {self.n_cv_splits} folds "
              f"(each val: 60 samples, NG=15, Good=45)")
        print(f"(k=15 선택, Task1/Task2 공식 기반 점수)")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")

            X_train_fold = X_train_hybrid[train_idx]
            y_train_fold = train_Y_series.values[train_idx]

            X_val = X_train_hybrid[val_idx]
            y_val = train_Y_series.values[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            # 기존 하이퍼파라미터 유지
            model = MainModel(n_estimators=200, random_state=42 + fold_idx, n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)

            val_prob_ng = model.predict_proba(X_val)[:, 1]

            # 공통 평가 함수 사용
            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
            )

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (k=15, NG15/Good45) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 9-2. Test와 유사한 Validation Set 평가 (별도 평가)
        test_similar_val_path = "../data/val_indices_test_similar.npy"
        if os.path.exists(test_similar_val_path):
            print("\n" + "=" * 80)
            print("[Main] Test와 유사한 Validation Set 평가 (별도 평가)")
            print("=" * 80)
            
            # 저장된 validation 인덱스 로드
            test_similar_val_indices = np.load(test_similar_val_path)
            print(f"[Main] 저장된 Validation 인덱스 로드: {test_similar_val_path}")
            print(f"  - Validation set 크기: {len(test_similar_val_indices)}개")
            print(f"  - NG: {(train_Y_series.values[test_similar_val_indices] == 1).sum()}개")
            print(f"  - Good: {(train_Y_series.values[test_similar_val_indices] == 0).sum()}개")
            
            # Train/Val 분리
            test_similar_train_indices = np.setdiff1d(
                np.arange(len(train_Y_series)), 
                test_similar_val_indices
            )
            
            X_train_test_similar = X_train_hybrid[test_similar_train_indices]
            y_train_test_similar = train_Y_series.values[test_similar_train_indices]
            
            X_val_test_similar = X_train_hybrid[test_similar_val_indices]
            y_val_test_similar = train_Y_series.values[test_similar_val_indices]
            
            print(f"[Main] Train size: {X_train_test_similar.shape[0]}")
            print(f"[Main] Val size  : {X_val_test_similar.shape[0]} "
                  f"(NG={ (y_val_test_similar==1).sum() }, Good={ (y_val_test_similar==0).sum() })")
            
            # RandomForest 학습 및 평가
            model_test_similar = MainModel(n_estimators=200, random_state=42, n_jobs=-1)
            model_test_similar.fit(X_train_test_similar, y_train_test_similar)
            
            val_prob_test_similar = model_test_similar.predict_proba(X_val_test_similar)[:, 1]
            
            # 평가 함수 사용
            roc_test_similar, profit_test_similar, score_test_similar = calculate_competition_score(
                y_true=y_val_test_similar,
                y_prob=val_prob_test_similar,
            )
            
            print("\n===== Test-Similar Validation Set 평가 결과 =====")
            print(f"ROC-AUC  : {roc_test_similar:.6f}")
            print(f"Profit   : {profit_test_similar:.2f}")
            print(f"Score    : {score_test_similar:.6f}")
            print("=" * 80)
        else:
            print(f"\n[Main] Test와 유사한 Validation Set 파일을 찾을 수 없습니다: {test_similar_val_path}")
            print(f"  -> analyze_test_similar_val.py를 먼저 실행하세요.")

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

        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:200]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:200]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/CNN_Extractor_RF_submission_{timestamp}.csv"

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
        n_epochs=12,
        batch_size=32,
        n_cv_splits=5,
        encoder_weight_path="../weight/best_model.pth"
    )
    submission_result = pipeline.run_production_pipeline()

    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()

