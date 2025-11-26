#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 모델 (CNN Feature + RandomForest) 성능 비교 파이프라인
=========================================================

핵심 아이디어:
1. (유지) E2E 모델을 학습시켜 2D 공간 패턴을 학습 (best_model.pth)
2. (변경) E2E 모델을 '피처 추출기'로만 사용 (마지막 head 레이어 제거)
3. (추가) 🔥 RandomForest 모델을 2가지 버전으로 학습 및 비교:
    - A: RandomForest + 기본 피처 (샘플 코드 방식)
    - B: RandomForest + 기본 피처 + CNN 피처 (하이브리드 방식)
4. (추가) 🔥 Validation Set (NG=15, Good=45)에서 '사진 속 최종 공식'으로 점수 비교
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier # 🔥 샘플 코드와 동일한 모델
from sklearn.metrics import roc_auc_score
# 🔥 추가 모델들
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("✅ CatBoost 사용 가능")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost가 설치되지 않았습니다. conda install -c conda-forge catboost")
import warnings
import matplotlib.pyplot as plt
import platform

# 1. 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Linux':
    plt.rcParams['font.family'] = 'NanumGothic'

# 2. 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 1. 데이터 전처리 클래스 (변경 없음)
# ----------------------------------------------------
class DataProcessor:
    """데이터 로딩 및 전처리 클래스"""
    
    def __init__(self):
        self.OE = None
        self.Scaler = None
        self.cat_list = None
        self.num_list = None
        self.x_min_global = None
        self.x_max_global = None
        self.y_min_global = None
        self.y_max_global = None
        self.basic_feature_dim = None 
    
    def load_data(self, train_path="data/train.csv", test_path="data/test.csv"): 
        """데이터 로딩"""
        self.train = pd.read_csv(train_path)
        # test.csv는 좌표 범위 분석(analyze_coordinate_range)을 위해서만 로드
        self.test = pd.read_csv(test_path) 
        
        print(f"Train shape: {self.train.shape}")
        print(f"Test shape: {self.test.shape}") 
        
        self.train_X_basic = self.train.drop(columns=['Class']).iloc[:,:-256*3]
        self.train_Y = self.train['Class'].apply(lambda x: 1 if x == 'NG' else 0) # NG=1
        
        print(f"Features shape (Train): {self.train_X_basic.shape}")
        print(f"Target distribution - Good: {(self.train_Y==0).sum()}, NG: {(self.train_Y==1).sum()}")
        
        return self.train, self.test, self.train_X_basic, self.train_Y
    
    def setup_basic_preprocessing(self, train_X_basic_df):
        """기본 피처 전처리 설정 (🔥 분할된 train set으로 fit)"""
        self.cat_list = train_X_basic_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.num_list = sorted(list(set(train_X_basic_df.columns) - set(self.cat_list)))
        
        self.OE = OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist', sparse_output=False)
        self.OE.fit(train_X_basic_df[self.cat_list])
        
        self.Scaler = StandardScaler()
        self.Scaler.fit(train_X_basic_df[self.num_list])
        
    def preprocess_basic(self, dataset):
        """기본 피처 전처리"""
        Xc = self.OE.transform(dataset[self.cat_list])
        Xn = self.Scaler.transform(dataset[self.num_list])
        combined = np.concatenate([Xc, Xn], axis=1)
        
        if self.basic_feature_dim is None:
            self.basic_feature_dim = combined.shape[1]
            print(f"기본 피처 차원: {self.basic_feature_dim}")
            
        return combined.astype(np.float32)
    
    def analyze_coordinate_range(self):
        """실제 x, y 좌표 범위 분석"""
        x_cols = [f'x{i}' for i in range(256)]
        y_cols = [f'y{i}' for i in range(256)]
        
        all_data = pd.concat([self.train, self.test], ignore_index=True) 
        x_values = all_data[x_cols].values.flatten()
        y_values = all_data[y_cols].values.flatten()
        
        x_values = x_values[~np.isnan(x_values)]
        y_values = y_values[~np.isnan(y_values)]
        
        self.x_min_global = x_values.min()
        self.x_max_global = x_values.max()
        self.y_min_global = y_values.min()
        self.y_max_global = y_values.max()
        
        print(f"📊 좌표 범위 분석 결과:")
        print(f"   X 좌표 범위: {self.x_min_global:.2f} ~ {self.x_max_global:.2f}")
        print(f"   Y 좌표 범위: {self.y_min_global:.2f} ~ {self.y_max_global:.2f}")
        
        return self.x_min_global, self.x_max_global, self.y_min_global, self.y_max_global

# ----------------------------------------------------
# 2. 래스터화 클래스 (변경 없음)
# ----------------------------------------------------
class SpatialRasterizer:
    def __init__(self, x_min, x_max, y_min, y_max, grid_size=64):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.grid_size = grid_size
        self.x_range = x_max - x_min if x_max > x_min else 1
        self.y_range = y_max - y_min if y_max > y_min else 1
        
    def rasterize_with_real_coordinates(self, data_row):
        x_cols = [f'x{i}' for i in range(256)]
        y_cols = [f'y{i}' for i in range(256)]
        p_cols = [f'p{i}' for i in range(256)]
        
        x_coords = data_row[x_cols].values
        y_coords = data_row[y_cols].values
        p_values = data_row[p_cols].values
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        count_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        for i in range(256):
            if not (np.isnan(x_coords[i]) or np.isnan(y_coords[i]) or np.isnan(p_values[i])):
                x_norm = (x_coords[i] - self.x_min) / self.x_range
                y_norm = (y_coords[i] - self.y_min) / self.y_range
                x_idx = int(np.clip(x_norm * (self.grid_size - 1), 0, self.grid_size - 1))
                y_idx = int(np.clip(y_norm * (self.grid_size - 1), 0, self.grid_size - 1))
                grid[y_idx, x_idx] += p_values[i]
                count_grid[y_idx, x_idx] += 1
        
        mask = count_grid > 0
        grid[mask] = grid[mask] / count_grid[mask]
        
        return grid
    
# ----------------------------------------------------
# 3. E2E 모델 정의 (변경 없음)
# ----------------------------------------------------

class ImageCNN(nn.Module):
    def __init__(self, output_dim=64, input_size=64):  # 🔥 output_dim 64→128, input_size 64→128
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)      # 🔥 kernel_size 3→5
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        final_size = input_size // 16
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.fc_out = nn.Linear(512, output_dim) 
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.batch_norm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.batch_norm3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.batch_norm4(self.conv4(x))), 2)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)


class FullE2EModel(nn.Module):
    def __init__(self, basic_feature_dim, image_cnn_output_dim=64, basic_mlp_output_dim=32, input_grid_size=64):  # 🔥 기본값 변경
        super(FullE2EModel, self).__init__()
        
        self.image_cnn = ImageCNN(output_dim=image_cnn_output_dim, input_size=input_grid_size)
        
        self.basic_mlp = nn.Sequential(
            nn.Linear(basic_feature_dim, basic_feature_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(basic_feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(basic_feature_dim * 2, basic_mlp_output_dim),
            nn.ReLU()
        )
        
        combined_dim = image_cnn_output_dim + basic_mlp_output_dim  # 128 + 32 = 160
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),  # 🔥 64→128로 증가
            nn.ReLU(),
            nn.BatchNorm1d(64),           # 🔥 64→128로 증가
            nn.Dropout(0.3),
            nn.Linear(64, 1)              # 🔥 64→128로 증가
        )
    
    def forward(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((img_feat, basic_feat), dim=1)
        output = self.head(combined) # 160차원 -> 1차원
        return output

    # 🔥 추가: 피처 추출을 위한 '머리 없는' forward
    def extract_features(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((img_feat, basic_feat), dim=1)
        return combined # 160차원 피처 반환


# ----------------------------------------------------
# 4. 데이터셋 클래스 (변경 없음)
# ----------------------------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, full_df, basic_features_np, rasterizer, labels_np=None):
        self.full_df = full_df.reset_index(drop=True)
        self.basic_features_np = basic_features_np
        self.rasterizer = rasterizer
        self.labels_np = labels_np
        self.is_test = (labels_np is None)

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        data_row = self.full_df.iloc[idx]
        image_grid = self.rasterizer.rasterize_with_real_coordinates(data_row)
        image_tensor = torch.from_numpy(image_grid).unsqueeze(0) # (1, 64, 64)
        basic_feat_tensor = torch.from_numpy(self.basic_features_np[idx])
        
        if self.is_test:
            # (이 코드는 test 예측을 안하므로 이 부분은 사용되지 않음)
            return image_tensor, basic_feat_tensor
        else:
            label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
            return image_tensor, basic_feat_tensor, label_tensor

# ----------------------------------------------------
# 5. 🔥 하이브리드 파이프라인 (15:45 샘플링, 최종 공식 적용)
# ----------------------------------------------------
class HybridModelPipeline:
    """하이브리드 (CNN + RandomForest) 비교 파이프라인"""
    
    def __init__(self, n_epochs=20, batch_size=32):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.cnn_model = None # E2E 모델
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 사용 디바이스: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
    
    def train_cnn_extractor(self, train_loader, val_loader):
        """E2E 모델을 '학습'시켜 피처 추출기(best_model.pth)를 만듭니다."""
        
        self.cnn_model = FullE2EModel(self.data_processor.basic_feature_dim, input_grid_size=64).to(self.device)
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss() 
        
        best_val_loss = np.inf
        print("\n🧠 1단계: CNN 피처 추출기 학습 시작...")
        
        for epoch in range(self.n_epochs):
            self.cnn_model.train()
            train_loss_total = 0.0
            
            for img, basic, labels in train_loader:
                img, basic, labels = img.to(self.device), basic.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.cnn_model(img, basic) # E2E 모델 학습
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item()
            
            # Validation
            self.cnn_model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for img, basic, labels in val_loader:
                    img, basic, labels = img.to(self.device), basic.to(self.device), labels.to(self.device)
                    outputs = self.cnn_model(img, basic)
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item()
            
            avg_train_loss = train_loss_total / len(train_loader)
            avg_val_loss = val_loss_total / len(val_loader)
            
            print(f"  Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.cnn_model.state_dict(), 'best_model.pth')
                print(f"     -> Best CNN Extractor saved with Val Loss: {best_val_loss:.4f}")
        print("✅ CNN 피처 추출기 학습 완료.")

    def extract_cnn_features(self, loader):
        """'학습된' E2E 모델을 사용해 'CNN 피처'를 추출합니다."""
        
        # 저장된 최고 성능 모델 불러오기
        if self.cnn_model is None:
            self.cnn_model = FullE2EModel(self.data_processor.basic_feature_dim, input_grid_size=64).to(self.device)
        self.cnn_model.load_state_dict(torch.load('best_model.pth'))
        self.cnn_model.eval()
        
        all_features = []
        with torch.no_grad():
            for img, basic, labels in loader: # 레이블은 사용 안함
                img, basic = img.to(self.device), basic.to(self.device)
                
                # 'head'를 제거하고 96차원 피처 추출
                features_batch = self.cnn_model.extract_features(img, basic)
                all_features.append(features_batch.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)

    def calculate_competition_score(self, y_true, y_prob):
        """
        🔥 (최종 수정) Validation Set에서 '사진 속 최종 공식'으로 점수 계산
        - y_true: 0(Good), 1(NG) (총 60개)
        - y_prob: NG(불량)일 확률 (0~1)
        """
        
        # 1. TASK 1: ROC-AUC Score
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # 2. TASK 2: Total Net Profit
        # k: 'decision=True'로 선택할 개수 (15개)
        k = 15 # 🔥 9 -> 15로 수정
            
        df_eval = pd.DataFrame({'prob': y_prob, 'true_label': y_true})
        
        # 'decision=True'인 Top k개 (불량률이 가장 낮은 k개) 선택
        selected_products_df = df_eval.nsmallest(k, 'prob')
        
        # 맞춘 Good(0) 개수
        correct_good_count = (selected_products_df['true_label'] == 0).sum()
        # 틀린 NG(1) 개수
        incorrect_ng_count = (selected_products_df['true_label'] == 1).sum()

        # Net Profit 계산
        total_net_profit = (100 * correct_good_count) - (150 * incorrect_ng_count)

        # 3. Final Total Score
        auc_comp = max(roc_auc - 0.5, 0) / 0.5
        profit_comp = max(total_net_profit, 0) / 1500  # 20000은 고정 스케일링 값
        
        total_score = np.sqrt(auc_comp * profit_comp)
        
        return total_score, roc_auc, total_net_profit, k


    def run_comparison_pipeline(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("🚀 하이브리드 (CNN+RF) vs 기본 (RF) 성능 비교 시작")
        print("=" * 60)
        
        # 1. 데이터 로딩
        print("\n📁 1단계: 데이터 로딩 (train.csv, test.csv)")
        train_df, test_df, train_X_basic_df, train_Y_series = \
            self.data_processor.load_data(train_path="./data/train.csv", test_path="./data/test.csv")
        
        # 2. 좌표 범위 분석
        print("\n📊 2단계: 좌표 범위 분석 (Train+Test 통합)")
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()
        
        # 3. 래스터화 설정
        print("\n🎯 3단계: 공간 래스터화 설정")
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)
        
        # 4. 🔥 Train / Validation 데이터 분리 (NG=15, Good=45)
        print("\n🔪 4단계: Train / Validation 데이터 분리 (NG=15, Good=45)")
        
        all_indices = train_Y_series.index
        all_labels = train_Y_series.values
        
        ng_indices = all_indices[all_labels == 1]
        good_indices = all_indices[all_labels == 0]
        
        # 🔥 9 -> 15로 수정
        val_ng_count = min(15, len(ng_indices))
        # 🔥 27 -> 45로 수정
        val_good_count = min(45, len(good_indices))
        
        if len(ng_indices) < 15 or len(good_indices) < 45:
            print(f"⚠️ 경고: 데이터 부족. NG {len(ng_indices)}개, Good {len(good_indices)}개.")
            print(f"   -> Val Set을 NG={val_ng_count}개, Good={val_good_count}개로 구성합니다.")

        np.random.seed(42) # 재현성을 위해
        val_ng_indices = np.random.choice(ng_indices, val_ng_count, replace=False)
        val_good_indices = np.random.choice(good_indices, val_good_count, replace=False)
        
        val_indices = np.concatenate([val_ng_indices, val_good_indices])
        
        # train_indices는 val_indices를 제외한 나머지
        train_indices = np.setdiff1d(all_indices, val_indices)
        
        
        # 기본 피처 (Pandas)
        train_X_basic_split_df = train_X_basic_df.iloc[train_indices]
        val_X_basic_split_df = train_X_basic_df.iloc[val_indices]
        
        # 전체 데이터 (Pandas)
        train_df_split = train_df.iloc[train_indices]
        val_df_split = train_df.iloc[val_indices]

        # 레이블 (Pandas Series)
        y_train_labels = train_Y_series.iloc[train_indices]
        y_val_labels = train_Y_series.iloc[val_indices]
        
        print(f"  Train set: {len(train_indices)}개, Validation set: {len(val_indices)}개")
        print(f"  (Val Set 구성: NG={val_ng_count}개, Good={val_good_count}개)")

        # 5. 기본 피처 전처리 (Numpy)
        print("\n🔄 5단계: 기본 피처 전처리 (Numpy 변환)")
        # 🔥 중요: .setup_basic_preprocessing을 train_set으로만 fit
        self.data_processor.setup_basic_preprocessing(train_X_basic_split_df) 
        
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_split_df)
        X_val_basic_np = self.data_processor.preprocess_basic(val_X_basic_split_df)
        
        # 6. 🔥 CNN 학습용 데이터셋/로더 생성
        print("\n📦 6단계: CNN 학습용 데이터셋/로더 생성")
        train_dataset = MultiModalDataset(train_df_split, X_train_basic_np, self.rasterizer, y_train_labels.values)
        val_dataset = MultiModalDataset(val_df_split, X_val_basic_np, self.rasterizer, y_val_labels.values)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 7. 🔥 CNN 피처 추출기 학습
        self.train_cnn_extractor(train_loader, val_loader)
        
        # 8. 🔥 CNN 피처 추출 (순서가 중요하므로 Shuffle=False)
        print("\n✨ 7단계: CNN 피처 추출 (Train/Val Set)")
        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader_seq = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        X_train_cnn_feats = self.extract_cnn_features(train_loader_seq)
        X_val_cnn_feats = self.extract_cnn_features(val_loader_seq)
        print(f"  추출된 CNN 피처 형태 (Train): {X_train_cnn_feats.shape}") 
        print(f"  추출된 CNN 피처 형태 (Val): {X_val_cnn_feats.shape}")   
        
        # 9. 🔥 하이브리드 피처 생성
        print("\n🧬 8단계: 하이브리드 피처 결합 (기본 + CNN)")
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_cnn_feats], axis=1)
        X_val_hybrid = np.concatenate([X_val_basic_np, X_val_cnn_feats], axis=1)
        print(f"  하이브리드 피처 형태 (Train): {X_train_hybrid.shape}") 
        
        # 10. 🔥 다중 모델 학습 및 성능 비교
        print("\n🤖 9단계: 다중 모델 학습 및 성능 비교")
        
        models_results = {}
        
        # 모델 1: RandomForest (기본)
        print("\n🌲 RandomForest 모델 학습...")
        rf_model = RandomForestClassifier(
            random_state=42, 
            n_estimators=300,        # 🔥 더 많은 트리
            max_depth=10,           # 🔥 깊이 제한
            min_samples_split=5,    # 🔥 분할 최소 샘플
            min_samples_leaf=2,     # 🔥 리프 최소 샘플
            max_features='sqrt',    # 🔥 피처 선택 방식
            bootstrap=True,         # 🔥 부트스트랩
            n_jobs=-1
        )
        rf_model.fit(X_train_hybrid, y_train_labels)
        rf_prob = rf_model.predict_proba(X_val_hybrid)[:, 1]
        rf_score, rf_auc, rf_profit, rf_k = self.calculate_competition_score(y_val_labels.values, rf_prob)
        models_results['RandomForest'] = {
            'model': rf_model, 'prob': rf_prob, 'score': rf_score, 
            'auc': rf_auc, 'profit': rf_profit, 'k': rf_k
        }
        
        # 모델 2: CatBoost (추천)
        if CATBOOST_AVAILABLE:
            print("\n🐱 CatBoost 모델 학습...")
            cat_model = CatBoostClassifier(
                random_seed=42, 
                iterations=300,          # 🔥 더 많은 반복
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=3,          # 🔥 정규화 추가
                bootstrap_type='Bernoulli',  # 🔥 부트스트랩 방식
                subsample=0.8,          # 🔥 샘플링 비율
                verbose=False,
                eval_metric='AUC',      # 🔥 AUC 최적화
                early_stopping_rounds=50  # 🔥 조기 종료
            )
            cat_model.fit(X_train_hybrid, y_train_labels)
            cat_prob = cat_model.predict_proba(X_val_hybrid)[:, 1]
            cat_score, cat_auc, cat_profit, cat_k = self.calculate_competition_score(y_val_labels.values, cat_prob)
            models_results['CatBoost'] = {
                'model': cat_model, 'prob': cat_prob, 'score': cat_score,
                'auc': cat_auc, 'profit': cat_profit, 'k': cat_k
            }
        else:
            print("⚠️ CatBoost를 사용할 수 없습니다. RandomForest만 사용합니다.")
        
        # 최고 성능 모델 선택
        best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['score'])
        best_result = models_results[best_model_name]

        # 11. 🔥 최종 결과 (다중 모델 비교)
        print("\n" + "=" * 80)
        print("🎉 다중 모델 성능 비교 결과 (하이브리드 피쳐 160차원)")
        print("=" * 80)
        print(f"  (Val Set: NG={val_ng_count}개, Good={val_good_count}개)")
        print(f"  (선택(k): 15개)")
        print("-" * 80)
        
        # 모든 모델 결과 출력
        for model_name, result in models_results.items():
            print(f"  📊 {model_name}:")
            print(f"    - Task 1 (ROC-AUC): {result['auc']:.4f}")
            print(f"    - Task 2 (Net Profit): {result['profit']:,.0f} 원")
            print(f"    - 🏆 최종 점수 (Total): {result['score']:.4f}")
            print("-" * 80)
        
        # 최고 성능 모델 강조
        print(f"🥇 최고 성능 모델: {best_model_name}")
        print(f"   최고 점수: {best_result['score']:.4f}")
        print("=" * 80)
        
        # 모델별 성능 순위
        sorted_models = sorted(models_results.items(), key=lambda x: x[1]['score'], reverse=True)
        print("📈 성능 순위:")
        for i, (model_name, result) in enumerate(sorted_models, 1):
            print(f"   {i}. {model_name}: {result['score']:.4f}")
        
        return models_results, best_model_name


def main():
    """메인 실행 함수"""
    pipeline = HybridModelPipeline(n_epochs=20, batch_size=32) # 에포크와 배치 크기 조절
    pipeline.run_comparison_pipeline()

if __name__ == "__main__":
    main()