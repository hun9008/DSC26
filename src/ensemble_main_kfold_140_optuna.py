# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import optuna  # Optuna 추가

from CNN_encoder import (
    DataProcessor,
    SpatialRasterizer,
    FeatureEncoder,
    MultiModalDataset,
)

# 평가 함수 공통 모듈
from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

from util.logger import TeeLogger


# ----------------------------------------------------
# 5. Main Model (여러 모델 앙상블 VotingClassifier 기반 이진 분류기)
# ----------------------------------------------------
class MainModel:
    """
    Main Model:
      - 입력: FeatureEncoder에서 추출한 feature + 기본 피처 (hybrid feature)
      - 모델: RF / ExtraTrees / GBM / HistGB / SVM 앙상블 (soft voting)
      - 출력: NG 확률 (클래스 1의 확률)
    """

    def __init__(self, params=None):
        if params is None:
            params = {}

        # RandomForest 하이퍼파라미터
        rf_n_estimators = params.get("rf_n_estimators", 500)
        rf_max_depth = params.get("rf_max_depth", 8)

        # ExtraTrees 하이퍼파라미터
        et_n_estimators = params.get("et_n_estimators", 500)
        et_max_depth = params.get("et_max_depth", None)

        # GradientBoosting 하이퍼파라미터
        gb_n_estimators = params.get("gb_n_estimators", 200)
        gb_learning_rate = params.get("gb_learning_rate", 0.05)
        gb_max_depth = params.get("gb_max_depth", 3)

        # HistGradientBoosting 하이퍼파라미터
        hist_max_depth = params.get("hist_max_depth", 10)
        hist_learning_rate = params.get("hist_learning_rate", 0.05)
        hist_max_iter = params.get("hist_max_iter", 300)

        # SVM 하이퍼파라미터
        svm_C = params.get("svm_C", 1.0)

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
            max_depth=hist_max_depth,
            learning_rate=hist_learning_rate,
            max_iter=hist_max_iter,
            random_state=42,
        )

        models["SVM"] = SVC(
            kernel="rbf",
            C=svm_C,
            gamma="scale",
            probability=True,  # soft voting에 필요
            random_state=42,
        )

        self.models = models

        estimators_for_voting = [(name, m) for name, m in self.models.items()]

        self.model = VotingClassifier(
            estimators=estimators_for_voting,
            voting="soft",  # 각 모델의 predict_proba를 평균
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 6. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """(사전 학습된) FeatureEncoder + MainModel 하이브리드 파이프라인"""

    def __init__(self, n_epochs=13, batch_size=32, n_cv_splits=5,
                 encoder_weight_path="feature_encoder.pth"):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None
        self.main_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path

    def load_pretrained_encoder(self, basic_feature_dim):
        if not os.path.exists(self.encoder_weight_path):
            raise FileNotFoundError(
                f"Pretrained encoder weight not found: {self.encoder_weight_path}\n"
                f"먼저 encoder_train.py 를 실행해서 feature_encoder.pth 를 생성하세요."
            )

        self.feature_encoder = FeatureEncoder(
            basic_feature_dim=basic_feature_dim
        ).to(self.device)

        state_dict = torch.load(self.encoder_weight_path, map_location=self.device)
        self.feature_encoder.load_state_dict(state_dict)
        self.feature_encoder.eval()
        print(f"[Main] Loaded encoder weights from {self.encoder_weight_path}")

    def extract_features(self, loader, is_test=False):
        assert self.feature_encoder is not None, "feature_encoder가 로드되지 않았습니다."
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

    # ---------------- Stratified K-Fold CV splits 생성 ----------------
    @staticmethod
    def make_cv_splits(y_series, n_splits=5, base_seed=42):
        """
        StratifiedKFold를 사용하여 label 비율을 유지한 채
        대략 len(y_series) / n_splits 개씩 validation을 만드는 함수.
        """
        y = y_series.values
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=base_seed
        )
        splits = []
        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            splits.append((train_idx, val_idx))
        return splits

    # ---------------- 전체 파이프라인 실행 ----------------
    def run_production_pipeline(self):

        logger = TeeLogger()
        sys.stdout = logger

        print("[Main] Start Production Pipeline (with Optuna tuning)")

        # 1. 데이터 로딩
        train_df, test_df, train_X_basic_df, train_Y_series, test_X_basic_df = \
            self.data_processor.load_data("../data/train.csv", "../data/test.csv")

        # 2. 좌표 범위 분석
        x_min, x_max, y_min, y_max = self.data_processor.analyze_coordinate_range()

        # 3. 래스터화 설정
        self.rasterizer = SpatialRasterizer(x_min, x_max, y_min, y_max, grid_size=64)

        # 4. 기본 피처 전처리
        self.data_processor.setup_basic_preprocessing(train_X_basic_df)
        X_train_basic_np = self.data_processor.preprocess_basic(train_X_basic_df)
        X_test_basic_np = self.data_processor.preprocess_basic(test_X_basic_df)

        print(f"[Main] Processed basic features (Train): {X_train_basic_np.shape}")
        print(f"[Main] Processed basic features (Test) : {X_test_basic_np.shape}")

        # 5. Dataset / DataLoader (feature 추출용)
        train_dataset = MultiModalDataset(
            train_df, X_train_basic_np, self.rasterizer, train_Y_series.values
        )
        test_dataset = MultiModalDataset(
            test_df, X_test_basic_np, self.rasterizer, labels_np=None
        )

        train_loader_seq = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 6. 사전 학습된 FeatureEncoder 로드
        self.load_pretrained_encoder(self.data_processor.basic_feature_dim)

        # 7. FeatureEncoder를 이용해 feature 추출
        X_train_feat = self.extract_features(train_loader_seq, is_test=False)
        X_test_feat = self.extract_features(test_loader, is_test=True)

        print(f"[Main] Encoded features (Train): {X_train_feat.shape}")
        print(f"[Main] Encoded features (Test) : {X_test_feat.shape}")

        # 8. 하이브리드 피처 생성
        X_train_hybrid = np.concatenate([X_train_basic_np, X_train_feat], axis=1)
        X_test_hybrid = np.concatenate([X_test_basic_np, X_test_feat], axis=1)

        print(f"[Main] Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"[Main] Hybrid features (Test) : {X_test_hybrid.shape}")

        # 9. Cross Validation splits (Stratified K-Fold)
        cv_splits = self.make_cv_splits(
            train_Y_series,
            n_splits=self.n_cv_splits,
            base_seed=42
        )

        # ---------------- Optuna 하이퍼파라미터 튜닝 ----------------
        def objective(trial):
            # 하이퍼파라미터 샘플링
            params = {
                "rf_n_estimators": trial.suggest_int("rf_n_estimators", 200, 800, step=100),
                "rf_max_depth": trial.suggest_int("rf_max_depth", 4, 12),
                "et_n_estimators": trial.suggest_int("et_n_estimators", 200, 800, step=100),
                "et_max_depth": trial.suggest_int("et_max_depth", 6, 20),
                "gb_n_estimators": trial.suggest_int("gb_n_estimators", 100, 400, step=50),
                "gb_learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True),
                "gb_max_depth": trial.suggest_int("gb_max_depth", 2, 5),
                "hist_max_depth": trial.suggest_int("hist_max_depth", 4, 16),
                "hist_learning_rate": trial.suggest_float("hist_learning_rate", 0.01, 0.2, log=True),
                "hist_max_iter": trial.suggest_int("hist_max_iter", 100, 500, step=50),
                "svm_C": trial.suggest_float("svm_C", 0.1, 10.0, log=True),
            }

            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_train_fold = X_train_hybrid[train_idx]
                y_train_fold = train_Y_series.values[train_idx]

                X_val = X_train_hybrid[val_idx]
                y_val = train_Y_series.values[val_idx]

                model = MainModel(params=params)
                model.fit(X_train_fold, y_train_fold)

                val_prob_ng = model.predict_proba(X_val)[:, 1]

                _, _, score = calculate_competition_score(
                    y_true=y_val,
                    y_prob=val_prob_ng,
                )

                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            print(f"[Optuna] Trial {trial.number} mean_score={mean_score:.6f}")
            return mean_score

        print("\n[Main] Start Optuna hyperparameter tuning...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)  # trial 수는 필요에 따라 조정

        print(f"[Main] Optuna best score : {study.best_value:.6f}")
        print(f"[Main] Optuna best params: {study.best_params}")

        best_params = study.best_params

        # ---------------- Best params로 CV 성능 다시 측정 (로그용) ----------------
        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []

        approx_val_size = len(train_Y_series) // self.n_cv_splits
        print(f"\n[Main] CV with best params (approx each val size: {approx_val_size})")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} (best params) =====")

            X_train_fold = X_train_hybrid[train_idx]
            y_train_fold = train_Y_series.values[train_idx]

            X_val = X_train_hybrid[val_idx]
            y_val = train_Y_series.values[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            model = MainModel(params=best_params)
            model.fit(X_train_fold, y_train_fold)

            val_prob_ng = model.predict_proba(X_val)[:, 1]

            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
            )

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (best params) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 10. 제출용 모델 재학습 (Train 전체 사용, best params)
        self.main_model = MainModel(params=best_params)
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

        # 여기서는 기존 로직 유지 (L/P 각각 170개 선택)
        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/ensemble_optuna_submission_{timestamp}.csv"

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
        n_epochs=13,
        batch_size=32,
        n_cv_splits=5,
        encoder_weight_path="../weight/feature_encoder.pth"
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()