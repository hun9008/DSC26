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
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from CNN_encoder import (
    DataProcessor,
    SpatialRasterizer,
    FeatureEncoder,
    MultiModalDataset,
)

# 평가 함수 공통 모듈
from util.eval import (
    evaluate_score_general,          # 안 쓰더라도 import 유지
    calculate_competition_score,
)

from util.logger import TeeLogger


# ----------------------------------------------------
# 1. Hill Climbing 기반 앙상블 가중치 탐색
# ----------------------------------------------------
def hill_climb_ensemble(
    y_true,
    pred_matrix,
    step=0.05,
    max_iter=200,
    min_weight=0.0,
    eps=1e-6,
):
    """
    y_true      : (N,) numpy array, 0/1 라벨
    pred_matrix : (N, M) numpy array, 각 열이 한 모델의 예측 확률
    return      : best_weights (M,), best_score
    """
    num_models = pred_matrix.shape[1]

    # 초기 가중치: 균등 분포
    w = np.ones(num_models, dtype=np.float64) / num_models

    def metric_for_w(w_vec):
        blended = pred_matrix @ w_vec
        _, _, score = calculate_competition_score(
            y_true=y_true,
            y_prob=blended,
        )
        return score

    best_score = metric_for_w(w)
    print(f"[HillClimb] init score = {best_score:.6f}, init w = {w}")

    for it in range(max_iter):
        improved = False
        best_local_w = w.copy()
        best_local_score = best_score

        for m in range(num_models):
            for delta in (+step, -step):
                w_new = w.copy()
                w_new[m] += delta

                # 최소 가중치 이하 or 합이 0 이하면 skip
                if w_new[m] < min_weight:
                    continue
                if w_new.sum() <= 0:
                    continue

                # 정규화
                w_new = w_new / w_new.sum()

                score_new = metric_for_w(w_new)

                if score_new > best_local_score + eps:
                    best_local_score = score_new
                    best_local_w = w_new
                    improved = True

        if not improved:
            print(f"[HillClimb] no improvement at iter {it}, stop.")
            break

        w = best_local_w
        best_score = best_local_score
        print(f"[HillClimb] iter {it+1}: score = {best_score:.6f}, w = {w}")

    print(f"[HillClimb] final score = {best_score:.6f}, final w = {w}")
    return w, best_score


# ----------------------------------------------------
# 2. Main Model (개별 모델 모음 - 20개 이상)
# ----------------------------------------------------
class MainModel:
    """
    Main Model:
      - 입력: FeatureEncoder에서 추출한 feature + 기본 피처 (hybrid feature)
      - 다양한 트리/부스팅/SVM/XGB/LGBM/CatBoost 모델 개별 학습
      - 출력: 각 모델의 NG 확률 (shape: [N, M])
    """

    def __init__(self):
        models = {}

        # ---------------- RandomForest (4개) ----------------
        models["RF_depth8_ne500"] = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        )
        # models["RF_depth10_ne400"] = RandomForestClassifier(
        #     n_estimators=400,
        #     max_depth=10,
        #     min_samples_leaf=2,
        #     max_features="sqrt",
        #     n_jobs=1,
        #     random_state=43,
        # )
        # models["RF_depthNone_ne300"] = RandomForestClassifier(
        #     n_estimators=300,
        #     max_depth=None,
        #     min_samples_leaf=4,
        #     max_features=0.7,
        #     n_jobs=1,
        #     random_state=44,
        # )
        # models["RF_depth6_ne600"] = RandomForestClassifier(
        #     n_estimators=600,
        #     max_depth=6,
        #     min_samples_leaf=1,
        #     max_features="log2",
        #     n_jobs=1,
        #     random_state=45,
        # )

        # ---------------- ExtraTrees (3개) ----------------
        models["ET_depthNone_ne500"] = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=1,
            random_state=52,
        )
        # models["ET_depth12_ne400"] = ExtraTreesClassifier(
        #     n_estimators=400,
        #     max_depth=12,
        #     min_samples_leaf=2,
        #     max_features=0.7,
        #     n_jobs=1,
        #     random_state=53,
        # )
        # models["ET_depth8_ne600"] = ExtraTreesClassifier(
        #     n_estimators=600,
        #     max_depth=8,
        #     min_samples_leaf=1,
        #     max_features="log2",
        #     n_jobs=1,
        #     random_state=54,
        # )

        # ---------------- GradientBoosting (3개) ----------------
        models["GB_lr005_depth3_ne200"] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=1.0,
            random_state=61,
        )
        # models["GB_lr01_depth3_ne150"] = GradientBoostingClassifier(
        #     n_estimators=150,
        #     learning_rate=0.1,
        #     max_depth=3,
        #     subsample=0.8,
        #     random_state=62,
        # )
        # models["GB_lr003_depth4_ne250"] = GradientBoostingClassifier(
        #     n_estimators=250,
        #     learning_rate=0.03,
        #     max_depth=4,
        #     subsample=0.9,
        #     random_state=63,
        # )

        # ---------------- HistGradientBoosting (3개) ----------------
        models["HGB_lr005_depth10_iter300"] = HistGradientBoostingClassifier(
            max_depth=10,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.0,
            random_state=71,
        )
        # models["HGB_lr01_depth8_iter200"] = HistGradientBoostingClassifier(
        #     max_depth=8,
        #     learning_rate=0.1,
        #     max_iter=200,
        #     l2_regularization=0.0,
        #     random_state=72,
        # )
        # models["HGB_lr003_depth12_iter400"] = HistGradientBoostingClassifier(
        #     max_depth=12,
        #     learning_rate=0.03,
        #     max_iter=400,
        #     l2_regularization=1e-4,
        #     random_state=73,
        # )

        # ---------------- SVM (4개) ----------------
        models["SVM_rbf_C1_gammaScale"] = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=81,
        )
        # models["SVM_rbf_C3_gammaScale"] = SVC(
        #     kernel="rbf",
        #     C=3.0,
        #     gamma="scale",
        #     probability=True,
        #     random_state=82,
        # )
        # models["SVM_rbf_C1_gammaAuto"] = SVC(
        #     kernel="rbf",
        #     C=1.0,
        #     gamma="auto",
        #     probability=True,
        #     random_state=83,
        # )
        models["SVM_poly_C1_deg3"] = SVC(
            kernel="poly",
            degree=3,
            C=1.0,
            gamma="scale",
            coef0=1.0,
            probability=True,
            random_state=84,
        )

        # ---------------- XGBoost (2개, CPU) ----------------
        models["XGB_hist_lr005_depth6_ne400"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",    # CPU hist
            n_jobs=1,
            random_state=91,
        )
        # models["XGB_hist_lr01_depth5_ne300"] = XGBClassifier(
        #     n_estimators=300,
        #     max_depth=5,
        #     learning_rate=0.1,
        #     subsample=0.9,
        #     colsample_bytree=0.8,
        #     objective="binary:logistic",
        #     eval_metric="logloss",
        #     tree_method="hist",
        #     n_jobs=1,
        #     random_state=92,
        # )

        # ---------------- CatBoost (2개, CPU) ----------------
        models["CatBoost_lr005_depth6_ne400"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=111,
            verbose=False,
            task_type="CPU",
            thread_count=1,
        )
        # models["CatBoost_lr01_depth4_ne300"] = CatBoostClassifier(
        #     iterations=300,
        #     depth=4,
        #     learning_rate=0.1,
        #     loss_function="Logloss",
        #     eval_metric="Logloss",
        #     random_seed=112,
        #     verbose=False,
        #     task_type="CPU",
        #     thread_count=1,
        # )

        # 1) AdaBoost (decision stump 기반, shallow tree boosting)
        models["AdaBoost_stump_ne300"] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=300,
            learning_rate=0.05,
            random_state=121,
        )

        # 2) Bagging + shallow tree (variance 줄이는 방향)
        models["Bagging_DT_depth5_ne200"] = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=5,
                random_state=131,
            ),
            n_estimators=200,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            n_jobs=1,
            random_state=132,
        )

        # 3) Bagging + SVC (kernel SVM을 조금 더 안정적으로)
        models["Bagging_SVC_rbf_C1"] = BaggingClassifier(
            estimator=SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=141,
            ),
            n_estimators=15,
            max_samples=0.7,
            max_features=0.7,
            bootstrap=True,
            n_jobs=1,
            random_state=142,
        )

        # 4) Logistic Regression (linear model, calibration 용도로 좋음)
        models["LogReg_l2_C1"] = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            n_jobs=1,
            random_state=151,
        )

        # 5) 작은 MLP (tabular + CNN feature 하이브리드에 잘 맞음)
        models["MLP_2layer_64_32"] = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=300,
            random_state=161,
        )

        # 총 모델 수 확인
        print(f"[MainModel] Total base models: {len(models)}")

        self.models = models
        self.model_names = list(models.keys())

    def fit_all(self, X, y):
        for name, model in self.models.items():
            print(f"[MainModel] Fitting {name} ...")
            model.fit(X, y)

    def predict_all_proba(self, X):
        """
        return: shape (N, M)
        각 column은 self.model_names 순서대로 NG(클래스1) 확률
        """
        prob_list = []
        for name in self.model_names:
            m = self.models[name]
            p = m.predict_proba(X)[:, 1]  # NG 확률
            prob_list.append(p.reshape(-1, 1))
        return np.concatenate(prob_list, axis=1)


# ----------------------------------------------------
# 3. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """(사전 학습된) FeatureEncoder + MainModel + Hill Climbing 앙상블 파이프라인"""

    def __init__(self, n_epochs=13, batch_size=32, n_cv_splits=5,
                 encoder_weight_path="feature_encoder.pth"):
        self.data_processor = DataProcessor()
        self.rasterizer = None
        self.feature_encoder = None
        self.main_model = None

        # GPU 6번 사용 (FeatureEncoder만)
        self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
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

        print("[Main] Start Production Pipeline")

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

        # 9. Cross Validation (Stratified K-Fold) - OOF 예측 생성
        cv_splits = self.make_cv_splits(
            train_Y_series,
            n_splits=self.n_cv_splits,
            base_seed=42
        )

        y_all = train_Y_series.values
        n_train = len(y_all)

        # 한 번 만들어서 모델 수 / 이름 파악
        tmp_model = MainModel()
        model_names = tmp_model.model_names
        num_models = len(model_names)
        print(f"[Main] Base models ({num_models}): {model_names}")

        # OOF / Test 예측 저장 공간
        oof_preds = np.zeros((n_train, num_models), dtype=np.float32)
        test_preds_folds = []  # 각 fold에서 (n_test, num_models)

        approx_val_size = n_train // self.n_cv_splits
        print(f"\n[Main] Cross Validation with {self.n_cv_splits} folds "
              f"(approx each val size: {approx_val_size})")
        print(f"(calculate_competition_score 사용, 기본 k=15)")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")

            X_train_fold = X_train_hybrid[train_idx]
            y_train_fold = y_all[train_idx]

            X_val = X_train_hybrid[val_idx]
            y_val = y_all[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            main_model = MainModel()
            main_model.fit_all(X_train_fold, y_train_fold)

            # 1) Val 예측 (각 모델)
            val_prob_matrix = main_model.predict_all_proba(X_val)  # (val, M)
            oof_preds[val_idx, :] = val_prob_matrix

            # 2) Test 예측 (각 모델)
            test_prob_matrix = main_model.predict_all_proba(X_test_hybrid)  # (test, M)
            test_preds_folds.append(test_prob_matrix)

            # Fold별 개별 모델 성능 찍기
            for m_idx, name in enumerate(model_names):
                val_prob_m = val_prob_matrix[:, m_idx]
                roc_m, profit_m, score_m = calculate_competition_score(
                    y_true=y_val,
                    y_prob=val_prob_m,
                )
                print(f"[Fold {fold_idx+1}] {name:25s} "
                      f"ROC={roc_m:.6f} Profit={profit_m:.2f} Score={score_m:.6f}")

        # 10. 전체 OOF 기준으로 개별 모델 성능
        print("\n===== OOF per-model performance =====")
        for m_idx, name in enumerate(model_names):
            prob_m = oof_preds[:, m_idx]
            roc_m, profit_m, score_m = calculate_competition_score(
                y_true=y_all,
                y_prob=prob_m,
            )
            print(f"[OOF] {name:25s} ROC={roc_m:.6f} Profit={profit_m:.2f} Score={score_m:.6f}")

        # 11. Hill Climbing으로 최적 가중치 찾기 (OOF 기준)
        best_w, best_score = hill_climb_ensemble(
            y_true=y_all,
            pred_matrix=oof_preds,
            step=0.05,
            max_iter=200,
            min_weight=0.0,
        )

        # 12. Hill Climbing 앙상블 OOF 성능 + Fold별 요약
        oof_blend = oof_preds @ best_w
        roc_all, profit_all, score_all = calculate_competition_score(
            y_true=y_all,
            y_prob=oof_blend,
        )
        print("\n===== OOF Hill-Climb Ensemble Performance =====")
        print(f"ROC-AUC = {roc_all:.6f}, Profit = {profit_all:.2f}, Score = {score_all:.6f}")
        print(f"Best weights:")
        for name, wv in zip(model_names, best_w):
            print(f"  {name:25s}: {wv:.4f}")

        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []
        for fold_idx, (_, val_idx) in enumerate(cv_splits):
            y_val = y_all[val_idx]
            val_prob = oof_blend[val_idx]
            roc_f, profit_f, score_f = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob,
            )
            cv_roc_list.append(roc_f)
            cv_profit_list.append(profit_f)
            cv_score_list.append(score_f)
            print(f"[Fold {fold_idx+1}] Blended ROC={roc_f:.6f} Profit={profit_f:.2f} Score={score_f:.6f}")

        print("\n===== CV Summary (Hill-Climb Ensemble) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 13. Test 예측: fold별 예측 평균 후 Hill Climb 가중치 적용
        test_preds_mean = np.mean(test_preds_folds, axis=0)  # (n_test, M)
        test_prob = test_preds_mean @ best_w
        print(f"\n[Main] Test prob range: {test_prob.min():.4f} ~ {test_prob.max():.4f}")

        # 14. 제출 파일 생성 (기존 로직 유지)
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
        save_path = f"../data/submission/ensemble_hillclimb_{timestamp}.csv"

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