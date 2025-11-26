import os
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP 스레드 수 제한(세그폴트 완화 목적)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)

import optuna  # Optuna for hyperparameter tuning


# =====================================================================
# 0. 평가 함수 (공식 스코어 함수 - 필요시 그대로 사용 가능)
# =====================================================================
def evaluate_score_general(
    y_ng,            # NG=1, Good=0
    prob_ng,         # NG일 확률 (predict_proba()[:,1])
    n_select_each=200,
    profit_good=100,
    cost_ng=150
):
    """대회에서 주어진 평가 스코어 계산 함수 (L/P 반으로 나누는 버전)"""
    y_ng = np.asarray(y_ng)
    prob_ng = np.asarray(prob_ng)
    n = len(y_ng)
    assert len(prob_ng) == n

    # Good을 1, NG를 0으로 변환
    y_good = 1 - y_ng
    prob_good = 1.0 - prob_ng

    eval_df = pd.DataFrame({
        "y_good": y_good,
        "prob_ng": prob_ng,
        "prob_good": prob_good
    })

    # L / P 반으로 나누어 의사결정
    half = n // 2
    eval_df["decision"] = False

    # n_select_each가 half보다 커도 iloc[:n_select_each]는 half까지만 선택되므로 안전
    top_L = eval_df.iloc[:half].sort_values("prob_ng").iloc[:n_select_each].index
    top_P = eval_df.iloc[half:].sort_values("prob_ng").iloc[:n_select_each].index

    eval_df.loc[top_L, "decision"] = True
    eval_df.loc[top_P, "decision"] = True

    # ROC-AUC (Good=1, prob_good 사용)
    roc_auc = roc_auc_score(eval_df["y_good"], eval_df["prob_good"])

    # 이익 계산
    is_decision = eval_df["decision"]
    is_good = eval_df["y_good"] == 1
    is_ng = eval_df["y_good"] == 0

    total_net_profit = (
        profit_good * (is_decision & is_good).sum()
        - cost_ng * (is_decision & is_ng).sum()
    )

    # 정규화
    part_auc = max(roc_auc - 0.5, 0) / 0.5

    n_decision = int(is_decision.sum())
    max_profit = profit_good * n_decision if n_decision > 0 else profit_good
    part_profit = max(total_net_profit, 0) / max_profit if max_profit > 0 else 0.0

    total_score = np.sqrt(part_auc * part_profit)

    print(f"ROC-AUC Score     : {roc_auc:.6f}")
    print(f"Total Net Profit  : {total_net_profit}")
    print(f"Final Total Score : {total_score:.6f}")

    return roc_auc, total_net_profit, total_score


# =====================================================================
# 1. 데이터 전처리
# =====================================================================
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

        print(f"Train shape: {self.train.shape}")
        print(f"Test shape : {self.test.shape}")
        print(f"Train basic features: {self.train_X_basic.shape}")
        print(f"Test basic features : {self.test_X_basic.shape}")
        print(f"Target distribution - Good: {(self.train_Y == 0).sum()}, NG: {(self.train_Y == 1).sum()}")

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
            print(f"Basic feature dim: {self.basic_feature_dim}")

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

        print(f"X range: {self.x_min_global:.2f} ~ {self.x_max_global:.2f}")
        print(f"Y range: {self.y_min_global:.2f} ~ {self.y_max_global:.2f}")

        return self.x_min_global, self.x_max_global, self.y_min_global, self.y_max_global


# =====================================================================
# 2. 좌표 래스터화
# =====================================================================
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


# =====================================================================
# 3. Feature Encoder (CNN + MLP)
# =====================================================================
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
      - concat 후 96차원 feature 반환(extract_features) 또는 head까지 통과시켜 logit 출력(forward)
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


# =====================================================================
# 4. Dataset
# =====================================================================
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


# =====================================================================
# 5. Main Model (Ensemble Voting: RF + ExtraTrees + GB + HistGB + SVM)
# =====================================================================
class MainModel:
    """
    Main Model:
      - 입력: FeatureEncoder에서 추출한 feature + 기본 피처
      - 모델: RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, SVM
      - VotingClassifier(soft voting)으로 앙상블
      - params dict를 통해 Optuna에서 샘플된 하이퍼파라미터를 반영
    """

    def __init__(self, params=None):
        if params is None:
            params = {}

        # RandomForest
        rf_n_estimators = params.get("rf_n_estimators", 500)
        rf_max_depth = params.get("rf_max_depth", 8)

        # ExtraTrees
        et_n_estimators = params.get("et_n_estimators", 500)
        et_max_depth = params.get("et_max_depth", None)

        # GradientBoosting
        gb_n_estimators = params.get("gb_n_estimators", 200)
        gb_learning_rate = params.get("gb_learning_rate", 0.05)
        gb_max_depth = params.get("gb_max_depth", 3)

        # HistGradientBoosting
        histgb_max_depth = params.get("histgb_max_depth", 10)
        histgb_learning_rate = params.get("histgb_learning_rate", 0.05)
        histgb_max_iter = params.get("histgb_max_iter", 300)

        # SVM
        svm_C = params.get("svm_C", 1.0)
        svm_gamma = params.get("svm_gamma", "scale")

        models = {}

        models["RandomForest"] = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            n_jobs=1,          # 안정성을 위해 1
            random_state=42,
        )

        models["ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=et_n_estimators,
            max_depth=et_max_depth,
            n_jobs=1,          # 안정성을 위해 1
            random_state=42,
        )

        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=gb_n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            random_state=42,
        )

        models["HistGB"] = HistGradientBoostingClassifier(
            max_depth=histgb_max_depth,
            learning_rate=histgb_learning_rate,
            max_iter=histgb_max_iter,
            random_state=42,
        )

        models["SVM"] = SVC(
            kernel="rbf",
            C=svm_C,
            gamma=svm_gamma,
            probability=True,  # soft voting에 필요
            random_state=42,
        )

        self.models = models

        estimators_for_voting = [(name, m) for name, m in self.models.items()]

        self.model = VotingClassifier(
            estimators=estimators_for_voting,
            voting="soft",  # 확률 평균
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# =====================================================================
# 6. 전체 파이프라인 + Cross Validation + Optuna
# =====================================================================
class ProductionPipeline:
    """FeatureEncoder + Ensemble MainModel 하이브리드 파이프라인"""

    def __init__(self, n_epochs=13, batch_size=32):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None
        self.main_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    # ---------------- Feature Encoder 학습 ----------------
    def train_feature_encoder(self, train_loader):
        self.feature_encoder = FeatureEncoder(
            basic_feature_dim=self.data_processor.basic_feature_dim
        ).to(self.device)

        optimizer = optim.Adam(self.feature_encoder.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.n_epochs):
            self.feature_encoder.train()
            train_loss_total = 0.0

            for img, basic, labels in train_loader:
                img = img.to(self.device)
                basic = basic.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.feature_encoder(img, basic)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item()

            avg_train_loss = train_loss_total / len(train_loader)
            print(f"Epoch [{epoch + 1}/{self.n_epochs}] Train Loss: {avg_train_loss:.4f}")

        torch.save(self.feature_encoder.state_dict(), 'feature_encoder.pth')

    def extract_features(self, loader, is_test=False):
        if self.feature_encoder is None:
            self.feature_encoder = FeatureEncoder(
                basic_feature_dim=self.data_processor.basic_feature_dim
            ).to(self.device)

        self.feature_encoder.load_state_dict(torch.load('feature_encoder.pth', map_location=self.device))
        self.feature_encoder.eval()

        all_features = []
        with torch.no_grad():
            if is_test:
                for img, basic in loader:
                    img = img.to(self.device)
                    basic = basic.to(self.device)
                    feats = self.feature_encoder.extract_features(img, basic)
                    all_features.append(feats.cpu().numpy())
            else:
                for img, basic, _ in loader:
                    img = img.to(self.device)
                    basic = basic.to(self.device)
                    feats = self.feature_encoder.extract_features(img, basic)
                    all_features.append(feats.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    # ---------------- Validation 인덱스 생성 (NG 15, Good 45) ----------------
    @staticmethod
    def make_fixed_validation_indices(y_series, n_ng_val=15, n_good_val=45, seed=42):
        """
        y_series: pandas Series, 값은 0(Good), 1(NG)
        반환: train_idx, val_idx (numpy array)
        """
        rng = np.random.RandomState(seed)
        y = np.asarray(y_series)
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

    # ---------------- 사진 속 최종 공식 기반 Validation 스코어 ----------------
    @staticmethod
    def calculate_competition_score(
        y_true,          # 0(Good), 1(NG)
        prob_ng,         # NG일 확률
        k=15,            # 선택할 개수
        profit_good=100,
        cost_ng=150
    ):
        """
        사진 속 최종 공식과 동일한 Validation 스코어 계산
        - ROC-AUC (y_true vs prob_ng)
        - Top-k (NG 확률이 낮은 순) 선택 후 Net Profit
        - Total Score = sqrt(auc_comp * profit_comp)
        """
        y_true = np.asarray(y_true)
        prob_ng = np.asarray(prob_ng)

        # 1) ROC-AUC (양성 클래스 = NG=1)
        roc_auc = roc_auc_score(y_true, prob_ng)

        # 2) Net Profit 계산
        df = pd.DataFrame({'prob_ng': prob_ng, 'label': y_true})
        selected = df.nsmallest(k, 'prob_ng')  # NG 확률이 낮은 = Good일 것 같은 상품

        correct_good = (selected['label'] == 0).sum()
        incorrect_ng = (selected['label'] == 1).sum()

        total_net_profit = profit_good * correct_good - cost_ng * incorrect_ng

        # 3) 최종 점수
        auc_comp = max(roc_auc - 0.5, 0.0) / 0.5
        max_profit = profit_good * k  # best case: k개 모두 Good
        profit_comp = max(total_net_profit, 0.0) / max_profit if max_profit > 0 else 0.0

        total_score = np.sqrt(auc_comp * profit_comp)

        return total_score, roc_auc, total_net_profit

    # ---------------- 일반 Cross Validation (베스트 파라미터 평가용) ----------------
    def cross_validate(self, X, y, params=None, n_folds=5, n_ng_val=15, n_good_val=45, base_seed=42):
        """
        params로 고정된 앙상블에 대한 CV 평가 (Optuna 튜닝 이후 확인용)
        - Validation: NG=15, Good=45
        - Score: 사진 속 최종 공식 (k=15)
        """
        fold_results = []

        print("\n" + "=" * 80)
        print("🎯 Cross Validation (사진 속 최종 공식 기반)")
        print("=" * 80)

        for fold in range(n_folds):
            seed = base_seed + fold
            train_idx, val_idx = self.make_fixed_validation_indices(
                y_series=y,
                n_ng_val=n_ng_val,
                n_good_val=n_good_val,
                seed=seed
            )

            X_train_fold = X[train_idx]
            y_train_fold = np.asarray(y)[train_idx]

            X_val_fold = X[val_idx]
            y_val_fold = np.asarray(y)[val_idx]

            print(f"\n----- Fold {fold + 1}/{n_folds} -----")
            print(f"Train size: {X_train_fold.shape[0]}, "
                  f"Val size: {X_val_fold.shape[0]} "
                  f"(NG={ (y_val_fold==1).sum() }, Good={ (y_val_fold==0).sum() })")

            model = MainModel(params=params)
            model.fit(X_train_fold, y_train_fold)

            val_prob_ng = model.predict_proba(X_val_fold)[:, 1]

            total_score, roc_auc, total_net_profit = self.calculate_competition_score(
                y_true=y_val_fold,
                prob_ng=val_prob_ng,
                k=15,
                profit_good=100,
                cost_ng=150
            )

            print(f"  ROC-AUC        : {roc_auc:.6f}")
            print(f"  Net Profit     : {total_net_profit:,.0f}")
            print(f"  Total Score    : {total_score:.6f}")

            fold_results.append((roc_auc, total_net_profit, total_score))

        roc_list = [r[0] for r in fold_results]
        profit_list = [r[1] for r in fold_results]
        score_list = [r[2] for r in fold_results]

        print("\n===== Cross Validation Summary (Ensemble, fixed params) =====")
        print(f"ROC-AUC  mean: {np.mean(roc_list):.6f}  std: {np.std(roc_list):.6f}")
        print(f"Profit   mean: {np.mean(profit_list):.2f}  std: {np.std(profit_list):.2f}")
        print(f"Score    mean: {np.mean(score_list):.6f}  std: {np.std(score_list):.6f}")

    # ---------------- Optuna Objective ----------------
    def optuna_objective(self, trial, X, y, n_folds=3, n_ng_val=15, n_good_val=45, base_seed=42):
        """
        Optuna trial에서 사용할 objective 함수
        - trial로부터 하이퍼파라미터를 샘플
        - 샘플된 params로 앙상블 모델 생성
        - n_folds번 반복 CV 수행 후 total_score 평균 리턴 (사진 속 공식)
        """

        # 1) 하이퍼파라미터 샘플링
        params = {
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 200, 800, step=100),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 4, 12),

            "et_n_estimators": trial.suggest_int("et_n_estimators", 200, 800, step=100),
            "et_max_depth": trial.suggest_int("et_max_depth", 4, 12),

            "gb_n_estimators": trial.suggest_int("gb_n_estimators", 100, 400, step=50),
            "gb_learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True),
            "gb_max_depth": trial.suggest_int("gb_max_depth", 2, 5),

            "histgb_max_depth": trial.suggest_int("histgb_max_depth", 4, 16),
            "histgb_learning_rate": trial.suggest_float("histgb_learning_rate", 0.01, 0.2, log=True),
            "histgb_max_iter": trial.suggest_int("histgb_max_iter", 100, 400, step=100),

            "svm_C": trial.suggest_float("svm_C", 0.1, 10.0, log=True),
            "svm_gamma": trial.suggest_float("svm_gamma", 1e-3, 1e-1, log=True),
        }

        fold_scores = []

        for fold in range(n_folds):
            seed = base_seed + fold
            train_idx, val_idx = self.make_fixed_validation_indices(
                y_series=y,
                n_ng_val=n_ng_val,
                n_good_val=n_good_val,
                seed=seed
            )

            X_train_fold = X[train_idx]
            y_train_fold = np.asarray(y)[train_idx]

            X_val_fold = X[val_idx]
            y_val_fold = np.asarray(y)[val_idx]

            model = MainModel(params=params)
            model.fit(X_train_fold, y_train_fold)
            val_prob_ng = model.predict_proba(X_val_fold)[:, 1]

            total_score, _, _ = self.calculate_competition_score(
                y_true=y_val_fold,
                prob_ng=val_prob_ng,
                k=15,
                profit_good=100,
                cost_ng=150
            )

            fold_scores.append(total_score)

        mean_score = float(np.mean(fold_scores))
        print(f"[Optuna Trial] mean total_score = {mean_score:.6f}")
        return mean_score  # maximize

    # ---------------- 전체 파이프라인 실행 ----------------
    def run_production_pipeline(self):
        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)

        # 4. 기본 피처 전처리 준비
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. Dataset / DataLoader
        train_dataset = MultiModalDataset(train_df, X_train_basic_np, self.rasterizer, train_Y_series.values)
        test_dataset = MultiModalDataset(test_df, X_test_basic_np, self.rasterizer, labels_np=None)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 6. Feature Encoder 학습
        self.train_feature_encoder(train_loader)

        # 7. Feature Encoder를 이용해 feature 추출
        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        X_train_feat = self.extract_features(train_loader_seq, is_test=False)
        X_test_feat = self.extract_features(test_loader, is_test=True)

        print(f"Encoded features (Train): {X_train_feat.shape}")
        print(f"Encoded features (Test) : {X_test_feat.shape}")

        # 8. 하이브리드 피처 생성 (기본 피처 + FeatureEncoder 피처)
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_feat], axis=1)
        X_test_hybrid = np.concatenate([X_test_basic_np, X_test_feat], axis=1)

        print(f"Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"Hybrid features (Test) : {X_test_hybrid.shape}")

        y_array = train_Y_series.values

        # 9. Optuna를 이용한 앙상블 하이퍼파라미터 튜닝
        print("\n===== Start Optuna Hyperparameter Tuning (Ensemble) =====")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.optuna_objective(
                trial,
                X=X_train_hybrid,
                y=y_array,
                n_folds=3,        # 속도 고려해서 3-fold tuning
                n_ng_val=15,
                n_good_val=45,
                base_seed=42
            ),
            n_trials=20,          # 시도 횟수
        )

        print("\n===== Optuna Best Result =====")
        print("Best total_score:", study.best_value)
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        best_params = study.best_params

        # 10. 베스트 파라미터로 CV 다시 수행 (5-fold)
        print("\n===== Cross Validation with Best Params (5-fold) =====")
        self.cross_validate(
            X=X_train_hybrid,
            y=y_array,
            params=best_params,
            n_folds=5,
            n_ng_val=15,
            n_good_val=45,
            base_seed=100
        )

        # 11. 최종 제출용 모델: 전체 train으로 재학습 (베스트 파라미터 사용)
        self.main_model = MainModel(params=best_params)
        self.main_model.fit(X_train_hybrid, y_array)

        # 12. Test 예측 (앙상블)
        test_prob = self.main_model.predict_proba(X_test_hybrid)[:, 1]
        print(f"\nTest prob range: {test_prob.min():.4f} ~ {test_prob.max():.4f}")

        # 13. 제출 파일 생성
        submission = pd.read_csv("../data/submission/sample_submission.csv")
        submission['probability'] = np.concatenate([test_prob, test_prob])
        submission['decision'] = False

        # L / P 구간별로 decision 선택 (공식 제출 규칙 유지)
        n_sub = len(submission)
        half_sub = n_sub // 2

        idx_L_sub = submission.index[:half_sub]
        idx_P_sub = submission.index[half_sub:]

        # 최종 제출에선 170개씩 선택하도록 유지
        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        submission.to_csv("../data/sample_submission_dummy/CNN_MLP_EnsembleVoting_OptunaCV_v2_submission.csv", index=False)
        print("Saved submission to ../data/sample_submission_dummy/CNN_MLP_EnsembleVoting_OptunaCV_v2_submission.csv")

        selected_count = submission['decision'].sum()
        print(f"Total selected products: {selected_count}")

        return submission


def main():
    pipeline = ProductionPipeline(n_epochs=13, batch_size=32)
    submission_result = pipeline.run_production_pipeline()

    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()