"""
2D CNN + RandomForest 파이프라인 (Adaptive Cropping 버전)
---------------------------------------------------------
evaluation_form.py의 파이프라인을 그대로 따라가되,
P_255_Mean_Diff.py에서 했던 것처럼 각 샘플의 x/y 범위에 맞춰 raster grid를 잘라낸다.
그 외 Task1/Task2/최종 점수 출력 포맷은 evaluation_form.py와 동일하게 유지한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

CATBOOST_AVAILABLE = False
if os.environ.get("ENABLE_CATBOOST", "0") == "1":
    try:
        from catboost import CatBoostClassifier  # type: ignore
        CATBOOST_AVAILABLE = True
        print("✅ CatBoost 활성화 (ENABLE_CATBOOST=1)")
    except Exception as exc:  # pragma: no cover - 진단 메시지
        CATBOOST_AVAILABLE = False
        print(f"⚠️ CatBoost 로드 실패: {exc}\n   -> RandomForest만 사용합니다.")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GRID_SIZE = 64
NUM_POINTS = 256
X_COLS = [f"x{i}" for i in range(NUM_POINTS)]
Y_COLS = [f"y{i}" for i in range(NUM_POINTS)]
P_COLS = [f"p{i}" for i in range(NUM_POINTS)]
WEIGHT_PATH = BASE_DIR / "best_model_adaptive.pth"


class DataProcessor:
    """evaluation_form.py와 동일한 데이터 로딩/전처리 클래스"""

    def __init__(self):
        self.OE = None
        self.Scaler = None
        self.cat_list = None
        self.num_list = None
        self.basic_feature_dim = None
        self.train = None
        self.test = None

    def load_data(self, train_path: str | Path | None = None, test_path: str | Path | None = None):
        train_path = Path(train_path) if train_path else DATA_DIR / "train.csv"
        test_path = Path(test_path) if test_path else DATA_DIR / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"train.csv를 찾을 수 없습니다: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"test.csv를 찾을 수 없습니다: {test_path}")

        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        train_X_basic = self.train.drop(columns=["Class"]).iloc[:, : -NUM_POINTS * 3]
        train_Y = self.train["Class"].apply(lambda x: 1 if x == "NG" else 0)
        return self.train, self.test, train_X_basic, train_Y

    def setup_basic_preprocessing(self, train_X_basic_df):
        self.cat_list = train_X_basic_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.num_list = sorted(list(set(train_X_basic_df.columns) - set(self.cat_list)))

        self.OE = OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist", sparse_output=False)
        self.OE.fit(train_X_basic_df[self.cat_list])

        self.Scaler = StandardScaler()
        self.Scaler.fit(train_X_basic_df[self.num_list])

    def preprocess_basic(self, dataset):
        if self.OE is None or self.Scaler is None:
            raise RuntimeError("setup_basic_preprocessing을 먼저 호출하세요.")

        Xc = self.OE.transform(dataset[self.cat_list])
        Xn = self.Scaler.transform(dataset[self.num_list])
        combined = np.concatenate([Xc, Xn], axis=1)

        if self.basic_feature_dim is None:
            self.basic_feature_dim = combined.shape[1]

        return combined.astype(np.float32)

    def analyze_coordinate_range(self):
        if self.train is None or self.test is None:
            raise RuntimeError("load_data 호출이 선행되어야 합니다.")

        concat_df = pd.concat([self.train, self.test], ignore_index=True)
        x_values = concat_df[X_COLS].values.flatten()
        y_values = concat_df[Y_COLS].values.flatten()

        x_values = x_values[~np.isnan(x_values)]
        y_values = y_values[~np.isnan(y_values)]

        self.x_min_global = x_values.min()
        self.x_max_global = x_values.max()
        self.y_min_global = y_values.min()
        self.y_max_global = y_values.max()
        return self.x_min_global, self.x_max_global, self.y_min_global, self.y_max_global


class AdaptiveSpatialRasterizer:
    """
    evaluation_form.py의 SpatialRasterizer를 변형하여,
    각 샘플의 x/y 범위에 맞춰 패딩 후 grid를 생성한다.
    """

    def __init__(self, x_min, x_max, y_min, y_max, grid_size=GRID_SIZE, pad_ratio=0.02, min_pad=1.0):
        self.global_bounds = (x_min, x_max, y_min, y_max)
        self.grid_size = grid_size
        self.pad_ratio = pad_ratio
        self.min_pad = min_pad

    def _local_bounds(self, coords: Iterable[float]):
        arr = np.asarray(coords, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        c_min = float(arr.min())
        c_max = float(arr.max())
        pad = max((c_max - c_min) * self.pad_ratio, self.min_pad)
        return c_min - pad, c_max + pad

    def rasterize_with_real_coordinates(self, data_row):
        x_coords = data_row[X_COLS].values.astype(np.float32)
        y_coords = data_row[Y_COLS].values.astype(np.float32)
        p_vals = data_row[P_COLS].values.astype(np.float32)

        x_bounds = self._local_bounds(x_coords)
        y_bounds = self._local_bounds(y_coords)

        if x_bounds is None or y_bounds is None:
            x_min, x_max, y_min, y_max = self.global_bounds
        else:
            x_min, x_max = x_bounds
            y_min, y_max = y_bounds

        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        count_grid = np.zeros_like(grid, dtype=np.int16)

        for x, y, p in zip(x_coords, y_coords, p_vals):
            if np.isnan(x) or np.isnan(y) or np.isnan(p):
                continue
            x_norm = (x - x_min) / x_range
            y_norm = (y - y_min) / y_range
            x_idx = int(np.clip(x_norm * (self.grid_size - 1), 0, self.grid_size - 1))
            y_idx = int(np.clip(y_norm * (self.grid_size - 1), 0, self.grid_size - 1))
            grid[y_idx, x_idx] += p
            count_grid[y_idx, x_idx] += 1

        mask = count_grid > 0
        grid[mask] = grid[mask] / count_grid[mask]
        return grid


class ImageCNN(nn.Module):
    def __init__(self, output_dim=64, input_size=GRID_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        final_size = input_size // 16
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.fc_out = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)


class FullE2EModel(nn.Module):
    def __init__(self, basic_feature_dim, image_cnn_output_dim=64, basic_mlp_output_dim=32, input_grid_size=GRID_SIZE):
        super().__init__()
        self.image_cnn = ImageCNN(output_dim=image_cnn_output_dim, input_size=input_grid_size)
        self.basic_mlp = nn.Sequential(
            nn.Linear(basic_feature_dim, basic_feature_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(basic_feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(basic_feature_dim * 2, basic_mlp_output_dim),
            nn.ReLU(),
        )
        combined_dim = image_cnn_output_dim + basic_mlp_output_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        combined = torch.cat((img_feat, basic_feat), dim=1)
        return self.head(combined)

    def extract_features(self, x_image, x_basic):
        img_feat = self.image_cnn(x_image)
        basic_feat = self.basic_mlp(x_basic)
        return torch.cat((img_feat, basic_feat), dim=1)


class MultiModalDataset(Dataset):
    def __init__(self, full_df, basic_features_np, rasterizer, labels_np=None):
        self.full_df = full_df.reset_index(drop=True)
        self.basic_features_np = basic_features_np
        self.rasterizer = rasterizer
        self.labels_np = labels_np
        self.is_test = labels_np is None

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        data_row = self.full_df.iloc[idx]
        image_grid = self.rasterizer.rasterize_with_real_coordinates(data_row)
        image_tensor = torch.from_numpy(image_grid).unsqueeze(0)
        basic_feat_tensor = torch.from_numpy(self.basic_features_np[idx])

        if self.is_test:
            return image_tensor, basic_feat_tensor

        label_tensor = torch.tensor(self.labels_np[idx], dtype=torch.float32).view(1)
        return image_tensor, basic_feat_tensor, label_tensor


def create_train_val_splits(train_df, train_X_basic_df, train_Y_series, val_ng=15, val_good=45, seed=42):
    """평가 설정과 동일하게 NG 15, Good 45개로 Validation을 고정 분할한다."""
    all_indices = train_Y_series.index
    all_labels = train_Y_series.values
    ng_indices = all_indices[all_labels == 1]
    good_indices = all_indices[all_labels == 0]

    val_ng_count = min(val_ng, len(ng_indices))
    val_good_count = min(val_good, len(good_indices))

    if len(ng_indices) < val_ng or len(good_indices) < val_good:
        print(f"⚠️ 경고: 데이터 부족. NG {len(ng_indices)}개, Good {len(good_indices)}개.")
        print(f"   -> Val Set을 NG={val_ng_count}개, Good={val_good_count}개로 구성합니다.")

    rng = np.random.default_rng(seed)
    val_ng_indices = rng.choice(ng_indices, val_ng_count, replace=False)
    val_good_indices = rng.choice(good_indices, val_good_count, replace=False)
    val_indices = np.concatenate([val_ng_indices, val_good_indices])
    train_indices = np.setdiff1d(all_indices, val_indices)

    return {
        "train_df": train_df.iloc[train_indices],
        "val_df": train_df.iloc[val_indices],
        "train_X_basic": train_X_basic_df.iloc[train_indices],
        "val_X_basic": train_X_basic_df.iloc[val_indices],
        "y_train": train_Y_series.iloc[train_indices],
        "y_val": train_Y_series.iloc[val_indices],
        "val_ng_count": val_ng_count,
        "val_good_count": val_good_count,
    }


class HybridModelPipeline:
    """evaluation_form.py 파이프라인을 그대로 가져와 AdaptiveRasterizer만 교체"""

    def __init__(self, n_epochs=20, batch_size=32):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.cnn_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def extract_cnn_features(self, loader):
        if self.cnn_model is None:
            self.cnn_model = FullE2EModel(self.data_processor.basic_feature_dim, input_grid_size=GRID_SIZE).to(self.device)
        if not WEIGHT_PATH.exists():
            raise FileNotFoundError(
                f"{WEIGHT_PATH.name} 파일이 존재하지 않습니다. "
                "먼저 train_only_cnn_extractor()를 실행해 CNN 가중치를 저장하세요."
            )
        state_dict = torch.load(WEIGHT_PATH, map_location=self.device)
        self.cnn_model.load_state_dict(state_dict)
        self.cnn_model.eval()

        all_features = []
        with torch.no_grad():
            for img, basic, _ in loader:
                img, basic = img.to(self.device), basic.to(self.device)
                features_batch = self.cnn_model.extract_features(img, basic)
                all_features.append(features_batch.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    @staticmethod
    def calculate_competition_score(y_true, y_prob):
        roc_auc = roc_auc_score(y_true, y_prob)
        k = 15
        df_eval = pd.DataFrame({"prob": y_prob, "true_label": y_true})
        selected = df_eval.nsmallest(k, "prob")
        correct_good = (selected["true_label"] == 0).sum()
        incorrect_ng = (selected["true_label"] == 1).sum()
        total_net_profit = (100 * correct_good) - (150 * incorrect_ng)
        auc_comp = max(roc_auc - 0.5, 0) / 0.5
        profit_comp = max(total_net_profit, 0) / 1500
        total_score = np.sqrt(auc_comp * profit_comp)
        return total_score, roc_auc, total_net_profit, k

    def run_comparison_pipeline(self):
        print("=" * 60)
        print("🚀 하이브리드 (CNN+RF) vs 기본 (RF) 성능 비교 시작")
        print("=" * 60)

        print("\n📁 1단계: 데이터 로딩 (train.csv, test.csv)")
        train_df, test_df, train_X_basic_df, train_Y_series = self.data_processor.load_data()

        print("\n📊 2단계: 좌표 범위 분석 (Train+Test 통합)")
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        print("\n🎯 3단계: 공간 래스터화 설정 (Adaptive Cropping)")
        self.rasterizer = AdaptiveSpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=GRID_SIZE)

        print("\n🔪 4단계: Train / Validation 데이터 분리 (NG=15, Good=45)")
        splits = create_train_val_splits(train_df, train_X_basic_df, train_Y_series)
        train_df_split = splits["train_df"]
        val_df_split = splits["val_df"]
        train_X_basic_split_df = splits["train_X_basic"]
        val_X_basic_split_df = splits["val_X_basic"]
        y_train_labels = splits["y_train"]
        y_val_labels = splits["y_val"]
        val_ng_count = splits["val_ng_count"]
        val_good_count = splits["val_good_count"]

        print(f"  Train set: {len(train_df_split)}개, Validation set: {len(val_df_split)}개")
        print(f"  (Val Set 구성: NG={val_ng_count}개, Good={val_good_count}개)")

        print("\n🔄 5단계: 기본 피처 전처리 (Numpy 변환)")
        self.data_processor.setup_basic_preprocessing(train_X_basic_split_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_split_df)
        X_val_basic_np = self.data_processor.preprocess_basic(val_X_basic_split_df)

        print("\n📦 6단계: CNN 입력 텐서 생성")
        train_dataset = MultiModalDataset(train_df_split, X_train_basic_np, self.rasterizer, y_train_labels.values)
        val_dataset = MultiModalDataset(val_df_split, X_val_basic_np, self.rasterizer, y_val_labels.values)

        print("\n✨ 7단계: CNN 피처 추출 (Train/Val Set)")
        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader_seq = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        X_train_cnn_feats = self.extract_cnn_features(train_loader_seq)
        X_val_cnn_feats = self.extract_cnn_features(val_loader_seq)
        print(f"  추출된 CNN 피처 형태 (Train): {X_train_cnn_feats.shape}")
        print(f"  추출된 CNN 피처 형태 (Val): {X_val_cnn_feats.shape}")

        print("\n🧬 8단계: 하이브리드 피처 결합 (기본 + CNN)")
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_cnn_feats], axis=1)
        X_val_hybrid = np.concatenate([X_val_basic_np, X_val_cnn_feats], axis=1)

        print("\n🤖 9단계: 다중 모델 학습 및 성능 비교")
        models_results = {}

        print("\n🌲 RandomForest 모델 학습...")
        rf_model = RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
        )
        rf_model.fit(X_train_hybrid, y_train_labels)
        rf_prob = rf_model.predict_proba(X_val_hybrid)[:, 1]
        rf_score, rf_auc, rf_profit, rf_k = self.calculate_competition_score(y_val_labels.values, rf_prob)
        models_results["RandomForest"] = {
            "model": rf_model,
            "prob": rf_prob,
            "score": rf_score,
            "auc": rf_auc,
            "profit": rf_profit,
            "k": rf_k,
        }

        if CATBOOST_AVAILABLE:
            print("\n🐱 CatBoost 모델 학습...")
            cat_model = CatBoostClassifier(
                random_seed=42,
                iterations=300,
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=3,
                bootstrap_type="Bernoulli",
                subsample=0.8,
                verbose=False,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            cat_model.fit(X_train_hybrid, y_train_labels)
            cat_prob = cat_model.predict_proba(X_val_hybrid)[:, 1]
            cat_score, cat_auc, cat_profit, cat_k = self.calculate_competition_score(y_val_labels.values, cat_prob)
            models_results["CatBoost"] = {
                "model": cat_model,
                "prob": cat_prob,
                "score": cat_score,
                "auc": cat_auc,
                "profit": cat_profit,
                "k": cat_k,
            }
        else:
            print("⚠️ CatBoost를 사용할 수 없습니다. RandomForest만 사용합니다.")

        best_model_name = max(models_results.keys(), key=lambda k: models_results[k]["score"])
        best_result = models_results[best_model_name]

        print("\n" + "=" * 80)
        print("🎉 다중 모델 성능 비교 결과 (Adaptive Raster + 하이브리드 피쳐)")
        print("=" * 80)
        print(f"  (Val Set: NG={val_ng_count}개, Good={val_good_count}개)")
        print(f"  (선택(k): {best_result['k']}개)")
        print("-" * 80)
        for model_name, result in models_results.items():
            print(f"  📊 {model_name}:")
            print(f"    - Task 1 (ROC-AUC): {result['auc']:.4f}")
            print(f"    - Task 2 (Net Profit): {result['profit']:,.0f} 원")
            print(f"    - 🏆 최종 점수 (Total): {result['score']:.4f}")
            print("-" * 80)
        print(f"🥇 최고 성능 모델: {best_model_name}")
        print(f"   최고 점수: {best_result['score']:.4f}")
        print("=" * 80)

        sorted_models = sorted(models_results.items(), key=lambda x: x[1]["score"], reverse=True)
        print("📈 성능 순위:")
        for i, (model_name, result) in enumerate(sorted_models, 1):
            print(f"   {i}. {model_name}: {result['score']:.4f}")

        return models_results, best_model_name


def train_only_cnn_extractor(n_epochs=50, batch_size=32, weight_path: Path = WEIGHT_PATH):
    """
    CNN 피처 추출기만 별도로 학습하고 가중치를 저장한다.
    저장된 weight는 HybridModelPipeline이 그대로 로드하여 사용한다.
    """
    print("=" * 60)
    print("🧠 CNN 피처 추출기 단독 학습 모드 시작")
    print("=" * 60)

    data_processor = DataProcessor()
    train_df, _, train_X_basic_df, train_Y_series = data_processor.load_data()
    x_min, x_max, y_min, y_max = data_processor.analyze_coordinate_range()
    rasterizer = AdaptiveSpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=GRID_SIZE)

    splits = create_train_val_splits(train_df, train_X_basic_df, train_Y_series)
    train_df_split = splits["train_df"]
    val_df_split = splits["val_df"]
    train_X_basic_split_df = splits["train_X_basic"]
    val_X_basic_split_df = splits["val_X_basic"]
    y_train_labels = splits["y_train"]
    y_val_labels = splits["y_val"]

    data_processor.setup_basic_preprocessing(train_X_basic_split_df)
    X_train_basic_np = data_processor.preprocess_basic(train_X_basic_split_df)
    X_val_basic_np = data_processor.preprocess_basic(val_X_basic_split_df)

    train_dataset = MultiModalDataset(train_df_split, X_train_basic_np, rasterizer, y_train_labels.values)
    val_dataset = MultiModalDataset(val_df_split, X_val_basic_np, rasterizer, y_val_labels.values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullE2EModel(data_processor.basic_feature_dim, input_grid_size=GRID_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = np.inf
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        train_loss_total = 0.0
        for img, basic, labels in train_loader:
            img, basic, labels = img.to(device), basic.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img, basic)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for img, basic, labels in val_loader:
                img, basic, labels = img.to(device), basic.to(device), labels.to(device)
                outputs = model(img, basic)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()

        avg_train_loss = train_loss_total / max(len(train_loader), 1)
        avg_val_loss = val_loss_total / max(len(val_loader), 1)
        print(f"[CNN Epoch {epoch+1}/{n_epochs}] Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), weight_path)
            print(f"   ↳ 새 최고 가중치 저장 ({best_val_loss:.4f}) -> {weight_path.name}")

    print("✅ CNN 단독 학습 완료. pipeline 실행 시 저장된 weight를 자동으로 로드합니다.")
    
def main_train_cnn_tuning():
    # 튜닝할 하이퍼파라미터 변경 예시: Epoch 50, 학습률 1e-4로 변경
    train_only_cnn_extractor(
        n_epochs=50,
        batch_size=32,
        weight_path=BASE_DIR / "best_model_adaptive.pth" # 저장 경로
    )

def main():
    pipeline = HybridModelPipeline(n_epochs=50, batch_size=32)
    pipeline.run_comparison_pipeline()

if __name__ == "__main__":
    main()