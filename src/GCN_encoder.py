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
# 2. 좌표 래스터화 (현재 GCN 버전에서는 사용 X, 인터페이스만 유지)
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
        # GCN 버전에서는 사용하지 않지만, 기존 호환성을 위해 남겨둠
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
# 3. GCN 기반 Point Encoder + FeatureEncoder
# ----------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        """
        x       : [B, N, in_dim]
        adj_norm: [B, N, N]  (정규화된 인접행렬)
        """
        h = self.linear(x)          # [B, N, out_dim]
        h = torch.bmm(adj_norm, h)  # [B, N, out_dim]
        return h


class PointGCN(nn.Module):
    """
    간단한 Point-GCN:
      - 입력: [B, N, 3]  (x, y, p)
      - (x, y) 기반 k-NN 그래프 구성
      - GCNLayer 여러 층 통과
      - global mean pooling으로 [B, out_dim] 그래프 임베딩
    """
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=64, num_layers=3, k=10):
        super().__init__()
        assert num_layers >= 2
        self.k = k

        layers = []
        layers.append(GCNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(GCNLayer(hidden_dim, hidden_dim))
        layers.append(GCNLayer(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def _build_adj(self, coords):
        """
        coords: [B, N, 3]  (x, y, p)
        x,y 만 사용해 k-NN 그래프 구성
        return: adj_norm [B, N, N]
        """
        B, N, _ = coords.shape
        device = coords.device

        # (x, y)만 사용
        xy = coords[..., :2]  # [B, N, 2]

        # pairwise distance: [B, N, N]
        dist = torch.cdist(xy, xy, p=2)

        # self 포함 가장 가까운 k+1개 중 자기 자신(0번째) 제외 → k개
        knn_idx = dist.topk(k=self.k + 1, largest=False).indices[:, :, 1:]  # [B, N, k]

        # adjacency 초기화
        adj = torch.zeros(B, N, N, device=device)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, self.k)
        node_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, -1, self.k)

        adj[batch_idx, node_idx, knn_idx] = 1.0
        # 대칭화
        adj = torch.maximum(adj, adj.transpose(1, 2))

        # self-loop 추가
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        adj = adj + eye

        # degree normalization: D^{-1/2} A D^{-1/2}
        deg = adj.sum(-1)  # [B, N]
        deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj_norm

    def forward(self, x):
        """
        x: [B, N, 3] (좌표 + 압력)
        """
        # NaN → 0
        x = torch.nan_to_num(x, nan=0.0)

        adj_norm = self._build_adj(x)  # [B, N, N]

        h = x
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer(h, adj_norm)
            if i != num_layers - 1:
                h = F.relu(h)

        # global mean pooling
        g = h.mean(dim=1)  # [B, out_dim]
        return g


class FeatureEncoder(nn.Module):
    """
    Feature Encoder (Point-GCN + MLP):
      - PointGCN: 256개 (x,y,p) 포인트 그래프 -> graph embedding
      - basic MLP: tabular 기본 피처 -> basic embedding
      - concat 후 head까지 통과시켜 logit 출력
    """

    def __init__(
        self,
        basic_feature_dim,
        graph_hidden_dim=64,
        graph_out_dim=64,
        basic_mlp_output_dim=32,
        input_grid_size=64  # 인터페이스 호환용(사용 X)
    ):
        super(FeatureEncoder, self).__init__()

        # 1) 포인트 기반 GCN 인코더
        self.point_gnn = PointGCN(
            in_dim=3,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            num_layers=3,
            k=10,
        )

        # 2) 기본 피처용 MLP
        self.basic_mlp = nn.Sequential(
            nn.Linear(basic_feature_dim, basic_feature_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(basic_feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(basic_feature_dim * 2, basic_mlp_output_dim),
            nn.ReLU()
        )

        # 3) 결합 후 head
        combined_dim = graph_out_dim + basic_mlp_output_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_points, x_basic):
        """
        x_points: [B, 256, 3] (GCN 입력)
        x_basic : [B, basic_dim] (tabular 피처)
        """
        graph_emb = self.point_gnn(x_points)    # [B, graph_out_dim]
        basic_feat = self.basic_mlp(x_basic)    # [B, basic_mlp_output_dim]
        combined = torch.cat((graph_emb, basic_feat), dim=1)
        output = self.head(combined)            # [B, 1]
        return output

    def extract_features(self, x_points, x_basic):
        """
        main_pipeline.py 에서 feature 추출용으로 사용
        """
        graph_emb = self.point_gnn(x_points)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((graph_emb, basic_feat), dim=1)
        return combined  # 예: graph_out_dim + basic_mlp_output_dim (기본 96차원)


# ----------------------------------------------------
# 4. Dataset (포인트 기반 GCN용)
# ----------------------------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, full_df, basic_features_np, rasterizer, labels_np=None):
        self.full_df = full_df.reset_index(drop=True)
        self.basic_features_np = basic_features_np
        self.rasterizer = rasterizer   # 현재는 사용하지 않지만 인터페이스 유지
        self.labels_np = labels_np
        self.is_test = (labels_np is None)

        # 미리 컬럼 이름 준비
        self.x_cols = [f'x{i}' for i in range(256)]
        self.y_cols = [f'y{i}' for i in range(256)]
        self.p_cols = [f'p{i}' for i in range(256)]

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        data_row = self.full_df.iloc[idx]

        x_coords = data_row[self.x_cols].values.astype(np.float32)
        y_coords = data_row[self.y_cols].values.astype(np.float32)
        p_values = data_row[self.p_cols].values.astype(np.float32)

        # NaN → 0
        x_coords = np.nan_to_num(x_coords, nan=0.0)
        y_coords = np.nan_to_num(y_coords, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=0.0)

        # [256, 3] 로 stack
        node_feats = np.stack([x_coords, y_coords, p_values], axis=1)  # (256, 3)
        node_tensor = torch.from_numpy(node_feats)                     # float32

        basic_feat_tensor = torch.from_numpy(self.basic_features_np[idx])

        if self.is_test:
            return node_tensor, basic_feat_tensor
        else:
            label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
            return node_tensor, basic_feat_tensor, label_tensor


# ----------------------------------------------------
# 5. FeatureEncoder 학습 루틴
# ----------------------------------------------------
def train_encoder(n_epochs=20, batch_size=32, save_dir="../weight"):

    logger = TeeLogger()
    sys.stdout = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EncoderTrain] Device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # 날짜 suffix 추가
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"GCN_feature_encoder_{date_str}.pth")

    dp = DataProcessor()
    train_df, test_df, train_X_df, train_Y, test_X_df = \
        dp.load_data("../data/train.csv", "../data/test.csv")

    # GCN 버전에서는 좌표 범위/래스터화는 사용하지 않지만, 기존 호출 흐름은 유지
    x_min, x_max, y_min, y_max = dp.analyze_coordinate_range()
    rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max)

    dp.setup_basic_preprocessing(train_X_df)
    X_train_basic = dp.preprocess_basic(train_X_df)

    dataset = MultiModalDataset(train_df, X_train_basic, rasterizer, train_Y.values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FeatureEncoder(dp.basic_feature_dim).to(device)

    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(n_epochs):
        model.train()
        loss_sum = 0.0

        for pts, basic, y in loader:
            pts = pts.to(device)      # [B, 256, 3]
            basic = basic.to(device)
            y = y.to(device)

            optim_.zero_grad()
            out = model(pts, basic)
            loss = criterion(out, y)
            loss.backward()
            optim_.step()
            loss_sum += loss.item()

        print(f"[EncoderTrain] Epoch {ep+1}/{n_epochs} Loss={loss_sum/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[EncoderTrain] Saved encoder weights to: {save_path}")

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"[Main] Log saved to: {logger.log_path}")


if __name__ == "__main__":
    train_encoder()