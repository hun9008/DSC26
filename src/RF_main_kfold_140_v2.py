# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from datetime import datetime

from CNN_encoder import (
    DataProcessor,
    SpatialRasterizer,
    FeatureEncoder,
    MultiModalDataset,
)

from util.eval_v2 import eval_official_on_probs
from util.logger import TeeLogger


# ----------------------------------------------------
# 1. Main Model (RandomForest 기반 이진 분류기)
# ----------------------------------------------------
class MainModel:
    def __init__(self, n_estimators=200, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 2. 전체 파이프라인
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

    @staticmethod
    def make_cv_splits(y_series, n_splits=5, base_seed=42):
        y = y_series
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

        y_all = train_Y_series.values
        n_all = len(y_all)

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
            train_df, X_train_basic_np, self.rasterizer, y_all
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

        # 8. 하이브리드 피처 생성 (지금은 feature만 사용)
        X_train_hybrid = X_train_feat
        X_test_hybrid = X_test_feat

        print(f"[Main] Hybrid features (Train): {X_train_hybrid.shape}")
        print(f"[Main] Hybrid features (Test) : {X_test_hybrid.shape}")

        # ------------------------------------------------
        # 9. train_sub / valid_holdout 분리 (예: 80/20)
        # ------------------------------------------------
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=1234
        )
        train_sub_idx, holdout_idx = next(sss.split(X_train_hybrid, y_all))

        X_sub = X_train_hybrid[train_sub_idx]
        y_sub = y_all[train_sub_idx]

        X_holdout = X_train_hybrid[holdout_idx]
        y_holdout = y_all[holdout_idx]

        print(f"\n[Split] train_sub size : {X_sub.shape[0]}")
        print(f"[Split] holdout size   : {X_holdout.shape[0]}")

        # ------------------------------------------------
        # 10. train_sub 에서만 K-fold OOF + CV 로그
        # ------------------------------------------------
        cv_splits = self.make_cv_splits(y_sub, n_splits=self.n_cv_splits, base_seed=42)

        oof_prob_sub = np.zeros(len(y_sub), dtype=np.float32)

        cv_roc_list = []
        cv_profit_list = []
        cv_score_list = []

        approx_val_size = len(y_sub) // self.n_cv_splits
        print(f"\n[Main] CV on train_sub with {self.n_cv_splits} folds "
              f"(approx each val size: {approx_val_size})")

        for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} on train_sub =====")

            X_tr = X_sub[tr_idx]
            y_tr = y_sub[tr_idx]

            X_val = X_sub[val_idx]
            y_val = y_sub[val_idx]

            print(f"[Main] Train_sub fold train size: {X_tr.shape[0]}")
            print(f"[Main] Train_sub fold val   size: {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            model = MainModel(
                n_estimators=200,
                random_state=42 + fold_idx,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)

            val_prob_ng = model.predict_proba(X_val)[:, 1]
            oof_prob_sub[val_idx] = val_prob_ng  # ★ OOF 채우기

            roc, profit, score = eval_official_on_probs(
                y_ng=y_val,
                prob_ng=val_prob_ng,
                max_select=200
            )

            print(f"  Fold ROC-AUC  : {roc:.6f}")
            print(f"  Fold Profit   : {profit}")
            print(f"  Fold Score    : {score:.6f}")

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary on train_sub =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # ------------------------------------------------
        # 11. train_sub 전체 OOF 기반 Official Score
        # ------------------------------------------------
        print("\n===== OOF-based Official Score on train_sub =====")
        roc_oof, profit_oof, score_oof = eval_official_on_probs(
            y_ng=y_sub,
            prob_ng=oof_prob_sub,
            max_select=200
        )
        print(f"OOF ROC-AUC     : {roc_oof:.6f}")
        print(f"OOF Net Profit  : {profit_oof}")
        print(f"OOF Total Score : {score_oof:.6f}")

        # ------------------------------------------------
        # 12. train_sub 전체로 학습 후, holdout 평가
        # ------------------------------------------------
        print("\n===== Train on train_sub, Eval on holdout =====")
        hold_model = MainModel(
            n_estimators=200,
            random_state=999,
            n_jobs=-1
        )
        hold_model.fit(X_sub, y_sub)
        hold_prob_ng = hold_model.predict_proba(X_holdout)[:, 1]

        roc_hold, profit_hold, score_hold = eval_official_on_probs(
            y_ng=y_holdout,
            prob_ng=hold_prob_ng,
            max_select=200
        )
        print(f"Holdout ROC-AUC     : {roc_hold:.6f}")
        print(f"Holdout Net Profit  : {profit_hold}")
        print(f"Holdout Total Score : {score_hold:.6f}")

        # ------------------------------------------------
        # 13. 최종: train 전체로 학습 후 test 예측 (submission)
        # ------------------------------------------------
        print("\n===== Train on FULL train, Predict TEST =====")
        self.main_model = MainModel(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        self.main_model.fit(X_train_hybrid, y_all)

        test_prob = self.main_model.predict_proba(X_test_hybrid)[:, 1]
        print(f"[Main] Test prob range: {test_prob.min():.4f} ~ {test_prob.max():.4f}")

        submission = pd.read_csv("../data/submission/sample_submission.csv")
        submission['probability'] = np.concatenate([test_prob, test_prob])
        submission['decision'] = False

        n_sub = len(submission)
        half_sub = n_sub // 2
        idx_L_sub = submission.index[:half_sub]
        idx_P_sub = submission.index[half_sub:]

        decision_id_L_list = submission.loc[idx_L_sub].sort_values(
            'probability', ascending=True
        ).iloc[:200]['ID']
        decision_id_P_list = submission.loc[idx_P_sub].sort_values(
            'probability', ascending=True
        ).iloc[:200]['ID']

        submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
        submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../data/submission/CNN_RF_submission_{timestamp}.csv"

        submission.to_csv(save_path, index=False)
        print(f"[Main] Saved submission to {save_path}")
        print(f"[Main] Total selected products: {submission['decision'].sum()}")

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

    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()