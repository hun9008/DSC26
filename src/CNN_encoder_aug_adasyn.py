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
from sklearn.model_selection import train_test_split
from datetime import datetime
from imblearn.over_sampling import ADASYN

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
        self.x_min_global = None
        self.x_max_global = None
        self.y_min_global = None
        self.y_max_global = None
        self.basic_feature_dim = None

    def load_data(self, train_path="../data/train.csv", test_path="../data/test.csv", val_indices=None):
        """
        Args:
            train_path: train 데이터 경로
            test_path: test 데이터 경로
            val_indices: Validation에 사용할 원본 데이터 인덱스 (None이면 전체 사용)
        """
        # 원본 데이터 저장 (증강 전)
        self.train_original = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        
        print(f"[EncoderTrain] 원본 데이터 개수: {len(self.train_original)}개")

        # Validation 인덱스가 주어진 경우, 해당 인덱스는 증강에서 제외
        if val_indices is not None:
            # 원본 데이터에서 validation 제외
            train_indices_for_aug = np.setdiff1d(np.arange(len(self.train_original)), val_indices)
            self.train_aug = self.train_original.iloc[train_indices_for_aug].copy()
            print(f"[EncoderTrain] Validation {len(val_indices)}개 제외, 증강용 데이터: {len(self.train_aug)}개 (예상: {len(self.train_original) - len(val_indices)}개)")
        else:
            # 전체 데이터로 증강
            self.train_aug = self.train_original.copy()
            print(f"[EncoderTrain] 전체 데이터 사용: {len(self.train_aug)}개")

        self.train_y_aug = self.train_aug['Class'].apply(lambda x: 1 if x == 'NG' else 0)
        print(f"[EncoderTrain] Target distribution before augmentation: Good: {(self.train_y_aug == 0).sum()}, NG: {(self.train_y_aug == 1).sum()}")
        self.train_X_aug = self.train_aug.drop(columns=['Class'])
        print(f"[EncoderTrain] Train X aug shape: {self.train_X_aug.shape}")

        # ADASYN   
        self.cat_list = self.train_X_aug.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.num_list = sorted(list(set(self.train_X_aug.columns) - set(self.cat_list)))
        
        print(f"[EncoderTrain] 범주형 컬럼: {len(self.cat_list)}개 - {self.cat_list}")
        print(f"[EncoderTrain] 수치형 컬럼: {len(self.num_list)}개")
        
        OE = OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist', sparse_output=False)
        OE.fit(self.train_X_aug[self.cat_list])
        Xc = pd.DataFrame(OE.transform(self.train_X_aug[self.cat_list]), columns=OE.get_feature_names_out(self.cat_list))
        Xn = self.train_X_aug[self.num_list]
        
        print(f"[EncoderTrain] OneHotEncoding 후 범주형 컬럼 수: {Xc.shape[1]}개")
        print(f"[EncoderTrain] 수치형 컬럼 수: {Xn.shape[1]}개")
        
        self.train_X_aug = pd.concat([Xc, Xn], axis=1)

        print(f"[EncoderTrain] ADASYN 적용 전 - Train X aug shape: {self.train_X_aug.shape}")
        print(f"[EncoderTrain] ADASYN 적용 전 - Train y aug shape: {self.train_y_aug.shape}")
        print(f"[EncoderTrain] ADASYN 적용 전 - Good: {(self.train_y_aug == 0).sum()}, NG: {(self.train_y_aug == 1).sum()}")
        
        self.adasyn = ADASYN(random_state=42)
        self.train_X_aug, self.train_y_aug = self.adasyn.fit_resample(self.train_X_aug, self.train_y_aug)
        
        print(f"[EncoderTrain] ADASYN 적용 후 - Train X aug shape: {self.train_X_aug.shape}")
        print(f"[EncoderTrain] ADASYN 적용 후 - Train y aug shape: {self.train_y_aug.shape}")
        print(f"[EncoderTrain] ADASYN 적용 후 - Good: {(self.train_y_aug == 0).sum()}, NG: {(self.train_y_aug == 1).sum()}")
        # train_X_aug는 numpy array이므로 DataFrame으로 변환 후 concat
        train_X_df = pd.DataFrame(self.train_X_aug)
        train_y_df = pd.DataFrame(self.train_y_aug, columns=['Class'])
        self.train = pd.concat([train_X_df.reset_index(drop=True), train_y_df], axis=1)

        # 좌표/압력 256*3 컬럼 제외한 기본 피처
        self.train_X_basic = self.train.drop(columns=['Class']).iloc[:, :-256*3]
        self.train_Y = self.train['Class']

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

    def analyze_coordinate_range(self):
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

        print(f"[EncoderTrain] X range: {self.x_min_global:.2f} ~ {self.x_max_global:.2f}")
        print(f"[EncoderTrain] Y range: {self.y_min_global:.2f} ~ {self.y_max_global:.2f}")

        return self.x_min_global, self.x_max_global, self.y_min_global, self.y_max_global


# ----------------------------------------------------
# 2. 좌표 래스터화
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
# 3. Feature Encoder (CNN + MLP)
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


class FeatureEncoder(nn.Module):
    """
    Feature Encoder:
      - ImageCNN: rasterized 좌표/압력 이미지 -> image embedding
      - basic MLP: tabular 기본 피처 -> basic embedding
      - concat 후 head까지 통과시켜 logit 출력
    """

    def __init__(self, basic_feature_dim, image_cnn_output_dim=64, basic_mlp_output_dim=32, input_grid_size=64):
        super(FeatureEncoder, self).__init__()

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
        output = self.head(combined)
        return output

    def extract_features(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((img_feat, basic_feat), dim=1)
        return combined  # 예: 96차원 feature


# ----------------------------------------------------
# 4. Dataset
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
        image_tensor = torch.from_numpy(image_grid).unsqueeze(0)  # (1, 64, 64)
        basic_feat_tensor = torch.from_numpy(self.basic_features_np[idx])

        if self.is_test:
            return image_tensor, basic_feat_tensor
        else:
            label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
            return image_tensor, basic_feat_tensor, label_tensor


# ----------------------------------------------------
# 5. FeatureEncoder 학습 루틴
# ----------------------------------------------------
def train_encoder(n_epochs=100, batch_size=32, save_dir="../weight", val_ratio=0.2, random_state=42):
    
    logger = TeeLogger()
    sys.stdout = logger

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EncoderTrain] Device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # 날짜 suffix 추가
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_save_path = os.path.join(save_dir, f"feature_encoder_aug_adasyn_best_{date_str}.pth")
    final_save_path = os.path.join(save_dir, f"feature_encoder_aug_adasyn_final_{date_str}.pth")
    val_indices_path = os.path.join(save_dir, f"val_indices_aug_adasyn_{date_str}.npy")

    # 원본 데이터 로드 (증강 전) - Validation split 생성용
    train_original = pd.read_csv("../data/train.csv")
    train_Y_original = train_original['Class'].apply(lambda x: 1 if x == 'NG' else 0)

    # 원본 데이터 기준으로 Train/Validation Split 생성
    train_idx_original, val_idx_original = train_test_split(
        np.arange(len(train_original)),
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_Y_original.values
    )
    
    print(f"[EncoderTrain] 원본 데이터 기준 Validation Split 생성")
    print(f"  - Train (증강용): {len(train_idx_original)}개")
    print(f"  - Val (증강 제외): {len(val_idx_original)}개")
    
    # Validation 인덱스 저장 (RF_main에서 사용)
    np.save(val_indices_path, val_idx_original)
    print(f"[EncoderTrain] Validation 인덱스 저장: {val_indices_path}")

    dp = DataProcessor()
    # Validation 인덱스를 제외한 데이터만 증강에 사용
    train_df, test_df, train_X_df, train_Y, test_X_df = \
        dp.load_data("../data/train.csv", "../data/test.csv", val_indices=val_idx_original)

    x_min, x_max, y_min, y_max = dp.analyze_coordinate_range()
    rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max)

    dp.setup_basic_preprocessing(train_X_df)
    X_train_basic = dp.preprocess_basic(train_X_df)
    
    train_df_split = train_df
    val_df_split = train_original.iloc[val_idx_original]
    X_train_basic_split = X_train_basic
    X_val_basic_split = dp.preprocess_basic(train_original.iloc[val_idx_original].drop(columns=['Class']))
    train_Y_split = train_Y
    val_Y_split = train_Y_original.iloc[val_idx_original]

    print(f"[EncoderTrain] Train size: {len(train_df_split)}")
    print(f"[EncoderTrain] Val size: {len(val_df_split)}")
    print(f"[EncoderTrain] Train distribution - Good: {(train_Y_split == 0).sum()}, NG: {(train_Y_split == 1).sum()}")
    print(f"[EncoderTrain] Val distribution - Good: {(val_Y_split == 0).sum()}, NG: {(val_Y_split == 1).sum()}")

    train_dataset = MultiModalDataset(train_df_split, X_train_basic_split, rasterizer, train_Y_split.values)
    val_dataset = MultiModalDataset(val_df_split, X_val_basic_split, rasterizer, val_Y_split.values)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FeatureEncoder(dp.basic_feature_dim).to(device)
    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Best epoch tracking
    best_val_loss = float('inf')
    best_epoch = 0

    # Phase 1: Train with validation to find best epoch
    print("\n===== Phase 1: Training with Validation =====")
    for ep in range(n_epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        for img, basic, y in train_loader:
            img = img.to(device)
            basic = basic.to(device)
            y = y.to(device)

            optim_.zero_grad()
            out = model(img, basic)
            loss = criterion(out, y)
            loss.backward()
            optim_.step()
            train_loss_sum += loss.item()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for img, basic, y in val_loader:
                img = img.to(device)
                basic = basic.to(device)
                y = y.to(device)

                out = model(img, basic)
                loss = criterion(out, y)
                val_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)

        # Check for best epoch
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = ep + 1
            torch.save(model.state_dict(), best_save_path)
            print(f"[EncoderTrain] Epoch {ep+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} *BEST*")
        else:
            print(f"[EncoderTrain] Epoch {ep+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print(f"\n[EncoderTrain] Best epoch: {best_epoch} with val loss: {best_val_loss:.4f}")
    print(f"[EncoderTrain] Best model saved to: {best_save_path}")

    # Phase 2: Retrain on full dataset for best_epoch epochs
    print("\n===== Phase 2: Retraining on Full Dataset =====")
    
    # Load best model weights
    model.load_state_dict(torch.load(best_save_path))
    
    # Use full dataset
    full_dataset = MultiModalDataset(train_df, X_train_basic, rasterizer, train_Y.values)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    print(f"[EncoderTrain] Retraining for {best_epoch} epochs on full dataset ({len(train_df)} samples)")

    for ep in range(best_epoch):
        model.train()
        loss_sum = 0.0

        for img, basic, y in full_loader:
            img = img.to(device)
            basic = basic.to(device)
            y = y.to(device)

            optim_.zero_grad()
            out = model(img, basic)
            loss = criterion(out, y)
            loss.backward()
            optim_.step()
            loss_sum += loss.item()

        print(f"[EncoderTrain] Epoch {ep+1}/{best_epoch} Loss={loss_sum/len(full_loader):.4f}")

    torch.save(model.state_dict(), final_save_path)
    print(f"[EncoderTrain] Final model saved to: {final_save_path}")

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"[Main] Log saved to: {logger.log_path}")
    
    # Validation 인덱스 경로 반환 (RF_main에서 사용)
    return val_indices_path, final_save_path

if __name__ == "__main__":
    train_encoder()