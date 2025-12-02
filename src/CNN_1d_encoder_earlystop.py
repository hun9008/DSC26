# encoder_train.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datetime import datetime

from util.logger import TeeLogger
import sys

# ----------------------------------------------------
# 1. 데이터 전처리
# ----------------------------------------------------
class DataProcessor:
    """데이터 로딩 및 전처리"""

    def __init__(self):
        self.OE = None
        self.Scaler = None
        self.cat_list = None
        self.num_list = None
        self.basic_feature_dim = None

    def load_data(self, train_path="../data/train.csv", test_path="../data/test.csv"):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        # 좌표/압력 256*3 컬럼 제외한 기본 피처
        self.train_X_basic = self.train.drop(columns=['Class']).iloc[:, :-256*3]
        self.train_Y = self.train['Class'].apply(lambda x: 1 if x == 'NG' else 0)

        self.test_X_basic = self.test.drop(columns=['ID']).iloc[:, :-256*3]

        print(f"[EncoderTrain] Train shape: {self.train.shape}")
        print(f"[EncoderTrain] Test shape : {self.test.shape}")
        print(f"[EncoderTrain] Train basic features: {self.train_X_basic.shape}")
        print(f"[EncoderTrain] Test basic features : {self.test_X_basic.shape}")
        print(f"[EncoderTrain] Target distribution - Good: {(self.train_Y == 0).sum()}, NG: {(self.train_Y == 1).sum()}")

        return self.train, self.test, self.train_X_basic, self.train_Y, self.test_X_basic

    def setup_basic_preprocessing(self, train_X_basic_df):
        self.cat_list = train_X_basic_df.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        self.num_list = sorted(list(set(train_X_basic_df.columns) - set(self.cat_list)))

        self.OE = OneHotEncoder(
            min_frequency=0.01,
            handle_unknown='infrequent_if_exist',
            sparse_output=False
        )
        if len(self.cat_list) > 0:
            self.OE.fit(train_X_basic_df[self.cat_list])
        else:
            # 범주형이 전혀 없을 때를 대비한 더미 fit
            self.OE.fit(pd.DataFrame(index=train_X_basic_df.index))

        self.Scaler = StandardScaler()
        self.Scaler.fit(train_X_basic_df[self.num_list])

    def preprocess_basic(self, dataset):
        if len(self.cat_list) > 0:
            Xc = self.OE.transform(dataset[self.cat_list])
        else:
            Xc = np.zeros((len(dataset), 0), dtype=np.float32)

        Xn = self.Scaler.transform(dataset[self.num_list])
        combined = np.concatenate([Xc, Xn], axis=1)

        if self.basic_feature_dim is None:
            self.basic_feature_dim = combined.shape[1]
            print(f"[EncoderTrain] Basic feature dim: {self.basic_feature_dim}")

        return combined.astype(np.float32)



# ----------------------------------------------------
# 2. PCA 기반 1D 변환 및 p값 정렬
# ----------------------------------------------------
class PCA1DProcessor:
    """XY 좌표에 PCA를 적용하여 1D로 변환하고, 정렬된 p값 반환"""
    
    def __init__(self):
        pass
    
    def process_to_1d(self, data_row):
        """
        각 샘플에 대해:
        1. XY 좌표 추출
        2. PCA로 1D 좌표 변환
        3. 1D 좌표 기준 정렬
        4. 정렬된 p값 반환 (256 길이)
        """
        x_cols = [f'x{i}' for i in range(256)]
        y_cols = [f'y{i}' for i in range(256)]
        p_cols = [f'p{i}' for i in range(256)]
        
        # numpy array로 명시적 변환 (pandas Series의 .values가 object dtype일 수 있음)
        x_coords = np.array(data_row[x_cols].values, dtype=np.float64)
        y_coords = np.array(data_row[y_cols].values, dtype=np.float64)
        p_values = np.array(data_row[p_cols].values, dtype=np.float64)
        
        # 유효한 포인트만 추출 (NaN이 아닌 것들)
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords) | np.isnan(p_values))
        
        if valid_mask.sum() == 0:
            # 유효한 포인트가 없으면 0으로 채운 배열 반환
            return np.zeros(256, dtype=np.float32)
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        p_valid = p_values[valid_mask]
        
        # (N, 2) 형태의 포인트 배열
        points = np.vstack([x_valid, y_valid]).T
        
        # PCA 1D projection
        pca = PCA(n_components=1)
        coord_1d = pca.fit_transform(points).ravel()
        
        # 1D 좌표 기준 정렬
        order = np.argsort(coord_1d)
        p_sorted = p_valid[order]

        p_1d = p_sorted.astype(np.float32)

        return p_1d


# ----------------------------------------------------
# 3. Feature Encoder (1D CNN + MLP)
# ----------------------------------------------------
class OneDCNN(nn.Module):
    """1D CNN for processing sorted p-values"""
    
    def __init__(self, output_dim=64, input_length=256):
        super(OneDCNN, self).__init__()
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        
        # Max pooling으로 길이 감소: 256 -> 128 -> 64 -> 32 -> 16
        final_length = input_length // 16
        self.fc1 = nn.Linear(256 * final_length, 512)
        self.fc_out = nn.Linear(512, output_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x shape: (batch, 1, 256)
        x = F.max_pool1d(F.relu(self.batch_norm1(self.conv1(x))), 2)
        x = F.max_pool1d(F.relu(self.batch_norm2(self.conv2(x))), 2)
        x = F.max_pool1d(F.relu(self.batch_norm3(self.conv3(x))), 2)
        x = F.max_pool1d(F.relu(self.batch_norm4(self.conv4(x))), 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)


class FeatureEncoder(nn.Module):
    """
    Feature Encoder:
      - OneDCNN: PCA로 정렬된 p값 시퀀스 -> 1D CNN embedding
      - basic MLP: tabular 기본 피처 -> basic embedding
      - concat 후 head까지 통과시켜 logit 출력
    """

    def __init__(self, basic_feature_dim, cnn_output_dim=64, basic_mlp_output_dim=64, input_length=256):
        super(FeatureEncoder, self).__init__()

        self.one_d_cnn = OneDCNN(output_dim=cnn_output_dim, input_length=input_length)

        self.basic_mlp = nn.Sequential(
            nn.Linear(basic_feature_dim, basic_feature_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(basic_feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(basic_feature_dim * 2, basic_mlp_output_dim),
            nn.ReLU()
        )

        combined_dim = cnn_output_dim + basic_mlp_output_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_1d, x_basic):
        cnn_feat = self.one_d_cnn(x_1d)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((cnn_feat, basic_feat), dim=1)
        output = self.head(combined)
        return output

    def extract_features(self, x_1d, x_basic):
        cnn_feat = self.one_d_cnn(x_1d)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((cnn_feat, basic_feat), dim=1)
        return combined  # 예: 96차원 feature


# ----------------------------------------------------
# 4. Dataset
# ----------------------------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, full_df, basic_features_np, pca_processor, labels_np=None):
        self.full_df = full_df.reset_index(drop=True)
        self.basic_features_np = basic_features_np
        self.pca_processor = pca_processor
        self.labels_np = labels_np
        self.is_test = (labels_np is None)

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        data_row = self.full_df.iloc[idx]
        p_1d = self.pca_processor.process_to_1d(data_row)  # (256,)
        p_1d_tensor = torch.from_numpy(p_1d).unsqueeze(0)  # (1, 256) for Conv1d
        basic_feat_tensor = torch.from_numpy(self.basic_features_np[idx])

        if self.is_test:
            return p_1d_tensor, basic_feat_tensor
        else:
            label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
            return p_1d_tensor, basic_feat_tensor, label_tensor


# ----------------------------------------------------
# 5. FeatureEncoder 학습 루틴
# ----------------------------------------------------
def train_encoder(
    n_epochs=100,
    batch_size=32,
    save_dir="../weight",
    patience=5,              # early stopping patience
    min_delta=1e-4            # 개선으로 인정할 최소 감소량
):
    logger = TeeLogger()
    sys.stdout = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EncoderTrain] Device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # 날짜 suffix 추가
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"feature_encoder_1dcnn_100_earlystop_{date_str}.pth")

    # -----------------------------
    # 1) 데이터 로딩 및 전처리 준비
    # -----------------------------
    dp = DataProcessor()
    train_df, test_df, train_X_df, train_Y, test_X_df = \
        dp.load_data("../data/train.csv", "../data/test.csv")

    # PCA 기반 1D 프로세서 초기화
    pca_processor = PCA1DProcessor()

    dp.setup_basic_preprocessing(train_X_df)
    X_all_basic = dp.preprocess_basic(train_X_df)
    y_all = train_Y.values.astype(np.float32)

    # -----------------------------
    # 2) Train / Validation split
    # -----------------------------
    indices = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_all,
        random_state=42
    )

    train_df_sub = train_df.iloc[train_idx].reset_index(drop=True)
    val_df_sub   = train_df.iloc[val_idx].reset_index(drop=True)

    X_train_basic = X_all_basic[train_idx]
    X_val_basic   = X_all_basic[val_idx]

    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]

    train_dataset = MultiModalDataset(train_df_sub, X_train_basic, pca_processor, y_train)
    val_dataset   = MultiModalDataset(val_df_sub,   X_val_basic,   pca_processor, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 3) 모델/옵티마이저/손실함수
    # -----------------------------
    model = FeatureEncoder(dp.basic_feature_dim).to(device)

    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    # -----------------------------
    # 4) 학습 루프 + Early Stopping
    # -----------------------------
    for ep in range(n_epochs):
        # ----- Train -----
        model.train()
        train_loss_sum = 0.0

        for p_1d, basic, y in train_loader:
            p_1d = p_1d.to(device)
            basic = basic.to(device)
            y = y.to(device)

            optim_.zero_grad()
            out = model(p_1d, basic)
            loss = criterion(out, y)
            loss.backward()
            optim_.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for p_1d, basic, y in val_loader:
                p_1d = p_1d.to(device)
                basic = basic.to(device)
                y = y.to(device)

                out = model(p_1d, basic)
                loss = criterion(out, y)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        print(
            f"[EncoderTrain] Epoch {ep+1}/{n_epochs} "
            f"TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f}"
        )

        # ----- Early Stopping 체크 -----
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
            print(f"[EncoderTrain]  ✓ Best val loss updated: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[EncoderTrain]  ↳ No improvement. patience={patience_counter}/{patience}")

            if patience_counter >= patience:
                print(
                    f"[EncoderTrain] Early stopping triggered at epoch {ep+1}. "
                    f"Best ValLoss={best_val_loss:.4f}"
                )
                break

    # -----------------------------
    # 5) Best weight 로 저장
    # -----------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), save_path)
    print(f"[EncoderTrain] Saved encoder weights to: {save_path}")

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"[Main] Log saved to: {logger.log_path}")

if __name__ == "__main__":
    train_encoder()