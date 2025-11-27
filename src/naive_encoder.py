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
from datetime import datetime

from util.logger import TeeLogger
import sys

# ----------------------------------------------------
# 1. 데이터 전처리 (x/y/p 포함 전부 탭уляр로 사용)
# ----------------------------------------------------
class DataProcessor:
    """데이터 로딩 및 전처리 (CNN/이미지 없이 전체 피처를 MLP 입력으로 사용)"""

    def __init__(self):
        self.OE = None
        self.Scaler = None
        self.cat_list = None
        self.num_list = None
        self.basic_feature_dim = None

    def load_data(self, train_path="../data/train.csv", test_path="../data/test.csv"):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        # Class / ID 만 제외하고 나머지 전부 사용 (기존 basic + x/y/p 256*3)
        self.train_X_basic = self.train.drop(columns=['Class'])
        self.train_Y = self.train['Class'].apply(lambda x: 1 if x == 'NG' else 0)

        self.test_X_basic = self.test.drop(columns=['ID'])

        print(f"[EncoderTrain] Train shape: {self.train.shape}")
        print(f"[EncoderTrain] Test shape : {self.test.shape}")
        print(f"[EncoderTrain] Train features (all) : {self.train_X_basic.shape}")
        print(f"[EncoderTrain] Test features  (all) : {self.test_X_basic.shape}")
        print(f"[EncoderTrain] Target distribution - Good: {(self.train_Y == 0).sum()}, NG: {(self.train_Y == 1).sum()}")

        return self.train, self.test, self.train_X_basic, self.train_Y, self.test_X_basic

    def setup_basic_preprocessing(self, train_X_basic_df):
        # 범주형 / 수치형 분리
        self.cat_list = train_X_basic_df.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        self.num_list = sorted(list(set(train_X_basic_df.columns) - set(self.cat_list)))

        # OneHotEncoder (범주형)
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

        # StandardScaler (수치형: 기본 + x/y/p 포함)
        self.Scaler = StandardScaler()
        self.Scaler.fit(train_X_basic_df[self.num_list])

    def preprocess_basic(self, dataset):
        # 범주형 인코딩
        if len(self.cat_list) > 0:
            Xc = self.OE.transform(dataset[self.cat_list])
        else:
            Xc = np.zeros((len(dataset), 0), dtype=np.float32)

        # 수치형 스케일링
        Xn = self.Scaler.transform(dataset[self.num_list])

        combined = np.concatenate([Xc, Xn], axis=1)

        if self.basic_feature_dim is None:
            self.basic_feature_dim = combined.shape[1]
            print(f"[EncoderTrain] Basic feature dim (all tabular incl. x/y/p): {self.basic_feature_dim}")

        return combined.astype(np.float32)


# ----------------------------------------------------
# 2. Feature Encoder (순수 MLP)
# ----------------------------------------------------
class FeatureEncoder(nn.Module):
    """
    Feature Encoder (MLP-only):
      - 입력: 전체 탭uliar 피처 (기본 + x/y/p)
      - 출력: 1차원 logit (NG=1, Good=0에 대한 로짓)
      - extract_features: 중간 임베딩 벡터 반환용 (원하면 downstream에서 사용)
    """

    def __init__(self, basic_feature_dim, hidden_dim1_factor=2, hidden_dim2=64):
        super(FeatureEncoder, self).__init__()

        hidden_dim1 = basic_feature_dim * hidden_dim1_factor

        self.mlp_feature = nn.Sequential(
            nn.Linear(basic_feature_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(0.3),
        )

        self.head = nn.Linear(hidden_dim2, 1)

    def forward(self, x_basic):
        feat = self.mlp_feature(x_basic)
        logit = self.head(feat)
        return logit

    def extract_features(self, x_basic):
        """
        추후 다른 모델에서 encoder로 사용할 때,
        이 함수를 통해 hidden representation(feat)를 추출.
        """
        feat = self.mlp_feature(x_basic)
        return feat  # 예: hidden_dim2 차원


# ----------------------------------------------------
# 3. Dataset (이미지 없이 탭uliar + 라벨만)
# ----------------------------------------------------
class MultiModalDataset(Dataset):
    """
    이제 이미지 사용 안 하므로:
      - basic_features_np: (N, D) numpy array
      - labels_np: (N,) or None (test)
    """

    def __init__(self, full_df, basic_features_np, labels_np=None):
        self.full_df = full_df.reset_index(drop=True)
        self.basic_features_np = basic_features_np
        self.labels_np = labels_np
        self.is_test = (labels_np is None)

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        basic_feat_tensor = torch.from_numpy(self.basic_features_np[idx])

        if self.is_test:
            return basic_feat_tensor
        else:
            label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
            return basic_feat_tensor, label_tensor


# ----------------------------------------------------
# 4. FeatureEncoder 학습 루틴 (MLP-only)
# ----------------------------------------------------
def train_encoder(n_epochs=1, batch_size=32, save_dir="../weight"):

    logger = TeeLogger()
    sys.stdout = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EncoderTrain] Device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # 날짜 suffix 추가
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"naive_feature_encoder_{date_str}.pth")

    # 1) 데이터 로딩
    dp = DataProcessor()
    train_df, test_df, train_X_df, train_Y, test_X_df = \
        dp.load_data("../data/train.csv", "../data/test.csv")

    # 2) 전처리 세팅 및 변환 (기본 + x/y/p 모두 포함)
    dp.setup_basic_preprocessing(train_X_df)
    X_train_basic = dp.preprocess_basic(train_X_df)

    print(f"[EncoderTrain] Preprocessed train features: {X_train_basic.shape}")

    # 3) Dataset / DataLoader
    dataset = MultiModalDataset(train_df, X_train_basic, train_Y.values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4) MLP Encoder 모델 정의
    model = FeatureEncoder(dp.basic_feature_dim).to(device)

    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # 5) 학습 루프
    for ep in range(n_epochs):
        model.train()
        loss_sum = 0.0

        for batch in loader:
            basic, y = batch
            basic = basic.to(device)
            y = y.to(device)

            optim_.zero_grad()
            out = model(basic)
            loss = criterion(out, y)
            loss.backward()
            optim_.step()
            loss_sum += loss.item()

        print(f"[EncoderTrain] Epoch {ep+1}/{n_epochs} Loss={loss_sum/len(loader):.4f}")

    # 6) 가중치 저장
    torch.save(model.state_dict(), save_path)
    print(f"[EncoderTrain] Saved encoder weights to: {save_path}")
    print(f"[EncoderTrain] Basic feature dim used: {dp.basic_feature_dim}")

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"[Main] Log saved to: {logger.log_path}")


if __name__ == "__main__":
    train_encoder()