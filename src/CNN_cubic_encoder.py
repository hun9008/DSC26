# CNN_cubic_encoder.py
# Cubic 보간법을 사용하는 래스터화 클래스
import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.interpolate import griddata

# ----------------------------------------------------
# 1. 데이터 전처리 클래스
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

    def load_data(self, train_path="../data/train.csv", test_path="../data/test.csv"):
        """데이터 로딩"""
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        print(f"Train shape: {self.train.shape}")
        print(f"Test shape: {self.test.shape}")

        self.train_X_basic = self.train.drop(columns=['Class']).iloc[:, :-256*3]
        self.train_Y = self.train['Class'].apply(lambda x: 1 if x == 'NG' else 0)  # NG=1

        self.test_X_basic = self.test.drop(columns=['ID']).iloc[:, :-256*3]

        print(f"Features shape (Train): {self.train_X_basic.shape}")
        print(f"Target distribution - Good: {(self.train_Y == 0).sum()}, NG: {(self.train_Y == 1).sum()}")

        return self.train, self.test, self.train_X_basic, self.train_Y, self.test_X_basic

    def setup_basic_preprocessing(self, train_X_basic_df):
        """기본 피처 전처리 설정 (분할된 train set으로 fit)"""
        self.cat_list = train_X_basic_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.num_list = sorted(list(set(train_X_basic_df.columns) - set(self.cat_list)))

        self.OE = OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist', sparse_output=False)
        if len(self.cat_list) > 0:
            self.OE.fit(train_X_basic_df[self.cat_list])
        else:
            # 범주형이 전혀 없을 때를 대비한 더미 fit
            self.OE.fit(pd.DataFrame(index=train_X_basic_df.index))

        self.Scaler = StandardScaler()
        self.Scaler.fit(train_X_basic_df[self.num_list])

    def preprocess_basic(self, dataset):
        """기본 피처 전처리"""
        if len(self.cat_list) > 0:
            Xc = self.OE.transform(dataset[self.cat_list])
        else:
            Xc = np.zeros((len(dataset), 0), dtype=np.float32)

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
# 2. 래스터화 클래스 (Cubic 보간법 사용)
# ----------------------------------------------------
class SpatialRasterizer:
    def __init__(self, x_min, x_max, y_min, y_max, grid_size=64, interpolation_method='linear'):
        """
        Args:
            x_min, x_max, y_min, y_max: 좌표 범위
            grid_size: 그리드 크기 (기본값: 64)
            interpolation_method: 보간 방법 ('linear' 또는 'cubic', 기본값: 'cubic')
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.grid_size = grid_size
        self.interpolation_method = interpolation_method
        self.x_range = x_max - x_min if x_max > x_min else 1
        self.y_range = y_max - y_min if y_max > y_min else 1

        print(f"📐 SpatialRasterizer 초기화:")
        print(f"   - 그리드 크기: {grid_size}x{grid_size}")
        print(f"   - 보간 방법: {interpolation_method}")

    def rasterize_with_real_coordinates(self, data_row, method=None):
        """
        선형/3차 보간법을 사용한 그리드 생성

        Args:
            data_row: 데이터 행 (Pandas Series)
            method: 보간 방법 ('linear' 또는 'cubic', None이면 self.interpolation_method 사용)

        Returns:
            grid: (grid_size, grid_size) 형태의 보간된 그리드
        """
        if method is None:
            method = self.interpolation_method

        x_cols = [f'x{i}' for i in range(256)]
        y_cols = [f'y{i}' for i in range(256)]
        p_cols = [f'p{i}' for i in range(256)]

        # pandas Series를 numpy array로 명시적 변환 (NaN 처리 포함)
        x_coords = pd.to_numeric(data_row[x_cols], errors='coerce').values
        y_coords = pd.to_numeric(data_row[y_cols], errors='coerce').values
        p_values = pd.to_numeric(data_row[p_cols], errors='coerce').values

        # numpy array로 명시적 변환
        x_coords = np.array(x_coords, dtype=np.float64)
        y_coords = np.array(y_coords, dtype=np.float64)
        p_values = np.array(p_values, dtype=np.float64)

        # 유효한 점들만 추출 (NaN이 아닌 점들)
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords) | np.isnan(p_values))

        if valid_mask.sum() < 3:  # 보간을 위해 최소 3개 점 필요
            # 점이 너무 적으면 0으로 채운 그리드 반환
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 유효한 점들의 좌표와 값
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_p = p_values[valid_mask]

        # 그리드 포인트 생성 (정규화된 좌표)
        x_grid = np.linspace(self.x_min, self.x_max, self.grid_size)
        y_grid = np.linspace(self.y_min, self.y_max, self.grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # 보간 수행
        # griddata는 (N,) 형태의 1D 배열을 받으므로 flatten
        grid_points = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
        data_points = np.column_stack([valid_x, valid_y])

        # scipy.interpolate.griddata를 사용한 보간
        # method: 'linear' (선형 보간) 또는 'cubic' (3차 보간)
        interpolated_values = griddata(
            data_points,           # 입력 점들의 좌표 (N, 2)
            valid_p,               # 입력 점들의 값 (N,)
            grid_points,           # 보간할 그리드 포인트들 (M, 2)
            method=method,         # 'linear' 또는 'cubic'
            fill_value=0.0         # 보간 불가능한 영역의 기본값
        )

        # 그리드 형태로 재구성
        grid = interpolated_values.reshape(self.grid_size, self.grid_size).astype(np.float32)

        return grid


# ----------------------------------------------------
# 3. 데이터셋 클래스
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

