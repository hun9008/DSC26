import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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
# 1. 데이터 전처리 클래스 (display.py와 동일)
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
    
    def load_data(self, train_path="train.csv", test_path="test.csv"): 
        """데이터 로딩"""
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        
        print(f"Train shape: {self.train.shape}")
        print(f"Test shape: {self.test.shape}") 
        
        self.train_X_basic = self.train.drop(columns=['Class']).iloc[:,:-256*3]
        self.train_Y = self.train['Class'].apply(lambda x: 1 if x == 'NG' else 0) # NG=1
        
        # 🔥 test 데이터 처리 추가
        self.test_X_basic = self.test.drop(columns=['ID']).iloc[:,:-256*3]
        
        print(f"Features shape (Train): {self.train_X_basic.shape}")
        print(f"Features shape (Test): {self.test_X_basic.shape}")
        print(f"Target distribution - Good: {(self.train_Y==0).sum()}, NG: {(self.train_Y==1).sum()}")
        
        return self.train, self.test, self.train_X_basic, self.train_Y, self.test_X_basic
    
    def setup_basic_preprocessing(self, train_X_basic_df):
        """기본 피처 전처리 설정 (전체 train set으로 fit)"""
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
# 2. 래스터화 클래스 (display.py와 동일)
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
# 3. E2E 모델 정의 (display.py와 동일)
# ----------------------------------------------------

class ImageCNN(nn.Module):
    def __init__(self, output_dim=64, input_size=64):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
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
    def __init__(self, basic_feature_dim, image_cnn_output_dim=64, basic_mlp_output_dim=32, input_grid_size=64):
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
        
        combined_dim = image_cnn_output_dim + basic_mlp_output_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1) 
        )
    
    def forward(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((img_feat, basic_feat), dim=1)
        output = self.head(combined) # 96차원 -> 1차원
        return output

    # 피처 추출을 위한 '머리 없는' forward
    def extract_features(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((img_feat, basic_feat), dim=1)
        return combined # 96차원 피처 반환


# ----------------------------------------------------
# 4. 데이터셋 클래스 (display.py와 동일)
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
            return image_tensor, basic_feat_tensor
        else:
            label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
            return image_tensor, basic_feat_tensor, label_tensor

# ----------------------------------------------------
# 5. 🔥 최종 제출 파이프라인 (train/validation 분리 없음)
# ----------------------------------------------------
class ProductionPipeline:
    """하이브리드 (CNN + RandomForest) 최종 제출 파이프라인"""
    
    def __init__(self, n_epochs=13, batch_size=32):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.cnn_model = None # E2E 모델
        self.rf_model = None  # RandomForest 모델
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 사용 디바이스: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
    
    def train_cnn_extractor(self, train_loader):
        """E2E 모델을 '학습'시켜 피처 추출기를 만듭니다."""
        
        self.cnn_model = FullE2EModel(self.data_processor.basic_feature_dim).to(self.device)
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss() 
        
        print("\n🧠 CNN 피처 추출기 학습 시작...")
        
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
            
            avg_train_loss = train_loss_total / len(train_loader)
            print(f"  Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # 최종 모델 저장
        torch.save(self.cnn_model.state_dict(), 'production_model.pth')
        print("✅ CNN 피처 추출기 학습 완료 및 저장.")

    def extract_cnn_features(self, loader, is_test=False):
        """'학습된' E2E 모델을 사용해 'CNN 피처'를 추출합니다."""
        
        # 저장된 모델 불러오기
        if self.cnn_model is None:
            self.cnn_model = FullE2EModel(self.data_processor.basic_feature_dim).to(self.device)
        self.cnn_model.load_state_dict(torch.load('production_model.pth'))
        self.cnn_model.eval()
        
        all_features = []
        with torch.no_grad():
            if is_test:
                for img, basic in loader: # test 데이터는 레이블 없음
                    img, basic = img.to(self.device), basic.to(self.device)
                    features_batch = self.cnn_model.extract_features(img, basic)
                    all_features.append(features_batch.cpu().numpy())
            else:
                for img, basic, labels in loader: # train 데이터는 레이블 있음
                    img, basic = img.to(self.device), basic.to(self.device)
                    features_batch = self.cnn_model.extract_features(img, basic)
                    all_features.append(features_batch.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)

    def run_production_pipeline(self):
        """최종 제출용 파이프라인 실행"""
        print("=" * 60)
        print("🚀 하이브리드 모델 최종 제출 파이프라인 시작")
        print("=" * 60)
        
        # 1. 데이터 로딩
        print("\n📁 1단계: 데이터 로딩 (train.csv, test.csv)")
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data(train_path="train.csv", test_path="test.csv")
        
        # 2. 좌표 범위 분석
        print("\n📊 2단계: 좌표 범위 분석 (Train+Test 통합)")
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()
        
        # 3. 래스터화 설정
        print("\n🎯 3단계: 공간 래스터화 설정")
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)
        
        # 4. 기본 피처 전처리 (전체 train 데이터로 fit)
        print("\n🔄 4단계: 기본 피처 전처리 (전체 train 데이터)")
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)
        
        print(f"  전처리된 기본 피처 형태 (Train): {X_train_basic_np.shape}")
        print(f"  전처리된 기본 피처 형태 (Test): {X_test_basic_np.shape}")
        
        # 5. CNN 학습용 데이터셋/로더 생성
        print("\n📦 5단계: CNN 학습용 데이터셋/로더 생성")
        train_dataset = MultiModalDataset(train_df, X_train_basic_np, self.rasterizer, train_Y_series.values)
        test_dataset = MultiModalDataset(test_df, X_test_basic_np, self.rasterizer, labels_np=None) # test는 레이블 없음
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 6. CNN 피처 추출기 학습
        print("\n🧠 6단계: CNN 피처 추출기 학습")
        self.train_cnn_extractor(train_loader)
        
        # 7. CNN 피처 추출 (train, test 모두)
        print("\n✨ 7단계: CNN 피처 추출 (Train/Test)")
        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        X_train_cnn_feats = self.extract_cnn_features(train_loader_seq, is_test=False)
        X_test_cnn_feats = self.extract_cnn_features(test_loader, is_test=True)
        
        print(f"  추출된 CNN 피처 형태 (Train): {X_train_cnn_feats.shape}") 
        print(f"  추출된 CNN 피처 형태 (Test): {X_test_cnn_feats.shape}")   
        
        # 8. 하이브리드 피처 생성
        print("\n🧬 8단계: 하이브리드 피처 결합 (기본 + CNN)")
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_cnn_feats], axis=1)
        X_test_hybrid = np.concatenate([X_test_basic_np, X_test_cnn_feats], axis=1)
        
        print(f"  하이브리드 피처 형태 (Train): {X_train_hybrid.shape}")
        print(f"  하이브리드 피처 형태 (Test): {X_test_hybrid.shape}")
        
        # 9. RandomForest 모델 학습
        print("\n🤖 9단계: RandomForest 모델 학습 (하이브리드 피처)")
        self.rf_model = RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1)
        self.rf_model.fit(X_train_hybrid, train_Y_series)
        print("✅ RandomForest 모델 학습 완료.")
        
        # 10. Test 데이터 예측
        print("\n🔮 10단계: Test 데이터 불량률 예측")
        test_prob = self.rf_model.predict_proba(X_test_hybrid)[:, 1]  # NG(1)일 확률
        print(f"  예측 완료: {len(test_prob)}개 샘플")
        print(f"  불량률 범위: {test_prob.min():.4f} ~ {test_prob.max():.4f}")
        
        # 11. 제출 파일 생성 (Sample_code.ipynb 양식)
        print("\n📝 11단계: 제출 파일 생성")
        submission = pd.read_csv("sample_submission.csv")
        
        # 🔥 Sample_code.ipynb와 동일한 방식으로 처리
        # test 데이터의 불량률을 L, P 각각에 복사 (concatenate)
        submission['probability'] = np.concatenate([test_prob, test_prob])
        
        # 불량률이 낮은 순서로 Top 200개씩 선택
        decision_id_L_list = submission.iloc[:466].sort_values('probability').iloc[:200]['ID']
        decision_id_P_list = submission.iloc[466:].sort_values('probability').iloc[:200]['ID']
        
        # decision=True로 설정
        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True
        
        # 제출 파일 저장
        submission.to_csv("hybrid_submission.csv", index=False)
        
        print("✅ 제출 파일 생성 완료: hybrid_submission.csv")
        print(f"   - L 타입에서 선택된 개수: {len(decision_id_L_list)}")
        print(f"   - P 타입에서 선택된 개수: {len(decision_id_P_list)}")
        print(f"   - 총 선택된 개수: {len(decision_id_L_list) + len(decision_id_P_list)}")
        
        # 12. 최종 결과 요약
        print("\n" + "=" * 60)
        print("🎉 하이브리드 모델 최종 제출 파이프라인 완료")
        print("=" * 60)
        print(f"  🔹 사용된 피처: 기본 피처 ({self.data_processor.basic_feature_dim}차원) + CNN 피처 (96차원)")
        print(f"  🔹 최종 피처 차원: {X_train_hybrid.shape[1]}차원")
        print(f"  🔹 학습 데이터: {len(train_Y_series)}개 (NG: {train_Y_series.sum()}개, Good: {len(train_Y_series)-train_Y_series.sum()}개)")
        print(f"  🔹 테스트 데이터: {len(test_prob)}개")
        print(f"  🔹 제출 파일: hybrid_submission.csv")
        print("=" * 60)
        
        return submission


def main():
    """메인 실행 함수"""
    pipeline = ProductionPipeline(n_epochs=13, batch_size=32)
    submission_result = pipeline.run_production_pipeline()
    
    # 결과 미리보기
    print("\n📋 제출 파일 미리보기:")
    print(submission_result.head(10))
    print("...")
    print(submission_result.tail(10))
    
    # 선택된 개수 확인
    selected_count = submission_result['decision'].sum()
    print(f"\n✅ 최종 선택된 제품 개수: {selected_count}개")

if __name__ == "__main__":
    main()
