# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier

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
# 5-1. RF Model
# ----------------------------------------------------
class RFModel:
    """
    RandomForest 기반 이진 분류기
    출력: predict_proba[:, 1] = NG(1) 확률
    """

    def __init__(self, n_estimators=200, random_state=42, n_jobs=-1,
                 max_depth=None, class_weight=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            max_depth=max_depth,
            class_weight=class_weight,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 5-2. CatBoost Model
# ----------------------------------------------------
class CatBoostModel:
    """
    CatBoost 기반 이진 분류기
    출력: predict_proba[:, 1] = NG(1) 확률
    """

    def __init__(
        self,
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        task_type=None,   # "GPU" 사용 시 지정
        devices=None,     # GPU id
    ):
        params = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_seed,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            eval_metric=eval_metric,
            verbose=verbose,
        )
        if task_type is not None:
            params["task_type"] = task_type
        if devices is not None:
            params["devices"] = devices

        self.model = CatBoostClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 5-3. TabPFN Model
# ----------------------------------------------------
class TabPFNModel:
    """
    TabPFN 기반 이진 분류기 (training-free meta-inference)
    출력: predict_proba[:, 1] = NG(1) 확률
    """

    def __init__(self, device=None, model_path=None, **kwargs):
        """
        device: "cpu" 또는 "cuda"
        model_path: TabPFN checkpoint 경로 (None이면 기본 모델 사용)
        kwargs: TabPFNClassifier 에 넘길 추가 인자
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path is not None:
            self.model = TabPFNClassifier(
                device=device,
                model_path=model_path,
                **kwargs,
            )
        else:
            self.model = TabPFNClassifier(
                device=device,
                **kwargs,
            )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model.fit(X, y)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)


# ----------------------------------------------------
# rank-based weighted averaging 유틸
# ----------------------------------------------------
def rank_weighted_ensemble(pred_list, weights=None):
    """
    pred_list: [p1, p2, p3, ...], 각 pi는 shape (N,) 의 확률 (예: NG 확률)
    weights:  [w1, w2, w3, ...] (None이면 모두 1)
    반환: shape (N,) 의 ensemble score (0~1로 정규화된 rank 기반 점수)

    - 각 모델마다 np.argsort(np.argsort(prob)) 로 rank 계산 (0 ~ N-1, 값 클수록 rank 큼)
    - 가중 평균 후 (N-1) 로 나눠서 0~1로 스케일
    """
    n_models = len(pred_list)
    if n_models == 0:
        raise ValueError("pred_list must contain at least one prediction array.")

    n_samples = len(pred_list[0])
    for p in pred_list:
        if len(p) != n_samples:
            raise ValueError("All prediction arrays must have the same length.")

    if weights is None:
        weights = np.ones(n_models, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape[0] != n_models:
            raise ValueError("weights length must match number of prediction arrays.")

    # 모델별 rank 계산
    rank_matrix = []
    for p in pred_list:
        # argsort(argsort)로 rank 계산 (0 = 가장 작은 확률, N-1 = 가장 큰 확률)
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_samples)
        rank_matrix.append(ranks.astype(np.float32))

    rank_matrix = np.stack(rank_matrix, axis=0)  # (M, N)

    # 가중 평균
    weighted_rank = np.average(rank_matrix, axis=0, weights=weights)

    # 0~1로 정규화
    denom = max(n_samples - 1, 1)
    ensemble_score = weighted_rank / denom
    return ensemble_score


# ----------------------------------------------------
# 6. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """
    (사전 학습된) FeatureEncoder + RF + CatBoost + TabPFN
    rank-based weighted ensemble 파이프라인
    """

    def __init__(
        self,
        n_epochs=13,
        batch_size=32,
        n_cv_splits=5,
        encoder_weight_path="feature_encoder.pth",
        # TabPFN 설정
        tabpfn_model_path=None,
        # 앙상블 가중치
        w_rf=1.0,
        w_cat=1.0,
        w_tab=1.0,
    ):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] Device: {self.device}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cv_splits = n_cv_splits
        self.encoder_weight_path = encoder_weight_path

        # TabPFN
        self.tabpfn_model_path = tabpfn_model_path

        # 앙상블 weight
        self.w_rf = w_rf
        self.w_cat = w_cat
        self.w_tab = w_tab

        # 최종 학습된 개별 모델들 (test 예측용)
        self.rf_model = None
        self.cat_model = None
        self.tab_model = None

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

        print("[Main] Start Production Pipeline (RF + CatBoost + TabPFN Ensemble)")

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

        # 8. 하이브리드 피처 생성 (필요하면 basic feature concat 가능)
        # X_train_hybrid = np.concatenate([X_train_basic_np, X_train_feat], axis=1)
        # X_test_hybrid = np.concatenate([X_test_basic_np, X_test_feat], axis=1)
        X_train_hybrid = X_train_feat.astype(np.float32)
        X_test_hybrid = X_test_feat.astype(np.float32)

        print(f"[Main] Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"[Main] Hybrid features (Test) : {X_test_hybrid.shape}")

        # 9. Cross Validation (Stratified K-Fold)
        cv_splits = self.make_cv_splits(
            train_Y_series,
            n_splits=self.n_cv_splits,
            base_seed=42
        )

        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []

        approx_val_size = len(train_Y_series) // self.n_cv_splits
        print(f"\n[Main] Cross Validation with {self.n_cv_splits} folds "
              f"(approx each val size: {approx_val_size})")
        print(f"(calculate_competition_score 사용, 기본 k=15)")
        print(f"[Main] Ensemble weights -> RF: {self.w_rf}, CatBoost: {self.w_cat}, TabPFN: {self.w_tab}")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")

            X_train_fold = X_train_hybrid[train_idx]
            y_train_fold = train_Y_series.values[train_idx]

            X_val = X_train_hybrid[val_idx]
            y_val = train_Y_series.values[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            # 9-1. RF 학습
            rf_model = RFModel(
                n_estimators=200,
                random_state=42 + fold_idx,
                n_jobs=-1,
                max_depth=None,
                class_weight=None,
            )
            rf_model.fit(X_train_fold, y_train_fold)
            val_prob_rf = rf_model.predict_proba(X_val)[:, 1]

            # 9-2. CatBoost 학습
            cat_model = CatBoostModel(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=42 + fold_idx,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False,
                # GPU 쓰고 싶으면 아래 두 줄 활성화
                # task_type="GPU",
                # devices="0",
            )
            cat_model.fit(X_train_fold, y_train_fold)
            val_prob_cat = cat_model.predict_proba(X_val)[:, 1]

            # 9-3. TabPFN 학습
            tab_model = TabPFNModel(
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_path=self.tabpfn_model_path,
            )
            tab_model.fit(X_train_fold, y_train_fold)
            val_prob_tab = tab_model.predict_proba(X_val)[:, 1]

            # 9-4. rank-based weighted ensemble
            val_prob_ens = rank_weighted_ensemble(
                [val_prob_rf, val_prob_cat, val_prob_tab],
                weights=[self.w_rf, self.w_cat, self.w_tab],
            )

            # 공통 평가 함수 사용 (k는 기본값 15 사용)
            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ens,
            )

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (StratifiedKFold, approx val~146) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 10. 제출용 모델 재학습 (Train 전체 사용)
        print("\n[Main] Train final models on full train set for submission")

        self.rf_model = RFModel(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            class_weight=None,
        )
        self.rf_model.fit(X_train_hybrid, train_Y_series.values)
        test_prob_rf = self.rf_model.predict_proba(X_test_hybrid)[:, 1]

        self.cat_model = CatBoostModel(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            # task_type="GPU",
            # devices="0",
        )
        self.cat_model.fit(X_train_hybrid, train_Y_series.values)
        test_prob_cat = self.cat_model.predict_proba(X_test_hybrid)[:, 1]

        self.tab_model = TabPFNModel(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_path=self.tabpfn_model_path,
        )
        self.tab_model.fit(X_train_hybrid, train_Y_series.values)
        test_prob_tab = self.tab_model.predict_proba(X_test_hybrid)[:, 1]

        # 11. Test rank ensemble
        test_prob_ens = rank_weighted_ensemble(
            [test_prob_rf, test_prob_cat, test_prob_tab],
            weights=[self.w_rf, self.w_cat, self.w_tab],
        )
        print(f"\n[Main] Test ensemble prob range: {test_prob_ens.min():.4f} ~ {test_prob_ens.max():.4f}")

        # 12. 제출 파일 생성
        submission = pd.read_csv("../data/submission/sample_submission.csv")
        submission['probability'] = np.concatenate([test_prob_ens, test_prob_ens])
        submission['decision'] = False

        n_sub = len(submission)
        half_sub = n_sub // 2

        idx_L_sub = submission.index[:half_sub]
        idx_P_sub = submission.index[half_sub:]

        # 기존 로직 유지 (L/P 각각 170개 선택, 확률 낮은 것부터)
        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:170]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/Ensemble_RF_Cat_TabPFN_submission_{timestamp}.csv"

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
        encoder_weight_path="../weight/feature_encoder.pth",
        tabpfn_model_path=None,   # 필요하면 TabPFN ckpt 경로 지정
        w_rf=1.0,
        w_cat=1.0,
        w_tab=1.0,
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()