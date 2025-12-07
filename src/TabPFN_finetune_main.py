# main_pipeline.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from functools import partial
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from sklearn.model_selection import StratifiedKFold, train_test_split

from tqdm import tqdm

from tabpfn import TabPFNClassifier  # TabPFN
from tabpfn.preprocessing import DatasetCollectionWithPreprocessing
from tabpfn.utils import meta_dataset_collator
from tabpfn.finetune_utils import clone_model_for_evaluation

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
# 5. Main Model (TabPFN 기반 + Full Fine-tuning)
# ----------------------------------------------------
class MainModel:
    """
    Main Model: TabPFNClassifier 기반 + (선택적) full fine-tuning
    """

    def __init__(
        self,
        device=None,
        model_path=None,
        do_finetune=True,
        finetune_epochs=1,
        finetune_lr=1e-5,
        max_data_size=150,
        n_estimators=2,
        **kwargs,
    ):
        """
        device: "cpu" 또는 "cuda"
        model_path: 오픈 버전 TabPFN ckpt 경로
                    예: "../weight/tabpfn_open.ckpt"
        do_finetune: True 이면 TabPFN backbone을 gradient로 파인튜닝
        finetune_epochs: 파인튜닝 epoch 수
        finetune_lr: AdamW learning rate
        max_data_size: TabPFN 한 번에 보는 최대 데이터 포인트 수
        n_estimators: TabPFN 앙상블 멤버 수
        kwargs: TabPFNClassifier 에 추가로 넘길 인자
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_path = model_path
        self.do_finetune = do_finetune
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.max_data_size = max_data_size
        self.n_estimators = n_estimators
        self.extra_kwargs = kwargs

        # TabPFN 기본 인자 세트 (평가용 / 클론용에서 재사용)
        base_args = {
            "device": self.device,
            "n_estimators": self.n_estimators,
        }
        if self.model_path is not None:
            base_args["model_path"] = self.model_path
        base_args.update(self.extra_kwargs)
        self.base_args = base_args

        # 최종 inference 에 사용할 모델 (fine-tune 후 clone된 모델)
        self.model = None

    # ---------- 내부 유틸: batched TabPFN 생성 ----------
    def _build_batched_classifier(self):
        """
        fine-tuning 용 TabPFNClassifier (fit_mode='batched') 생성
        """
        clf = TabPFNClassifier(
            **self.base_args,
            fit_mode="batched",  # batched 모드 활성화 (fine-tune용)
        )
        return clf

    # ---------- 내부 유틸: fine-tune dataset/dataloader ----------
    def _build_finetune_dataloader(self, X_train, y_train):
        """
        TabPFN fine-tuning을 위한 DatasetCollectionWithPreprocessing + DataLoader 생성
        X_train: numpy array, shape (N, D)
        y_train: numpy array, shape (N,)
        """
        # TabPFN이 in-context learning을 하므로, 내부적으로 train/test split 함수가 필요
        # (여기서는 label 분포를 유지하도록 stratify 사용)
        split_fn = partial(
            train_test_split,
            test_size=0.2,
            stratify=y_train,
        )

        clf = self._build_batched_classifier()

        datasets_collection: DatasetCollectionWithPreprocessing = clf.get_preprocessed_datasets(
            X_train,
            y_train,
            split_fn,
            max_data_size=self.max_data_size,
        )

        data_loader = DataLoader(
            datasets_collection,
            batch_size=1,                  # 현재는 1만 지원
            collate_fn=meta_dataset_collator,
        )

        return clf, data_loader

    # ---------- 내부 유틸: fine-tuning 루프 ----------
    def _run_finetuning(self, clf, data_loader):
        """
        Medium 튜토리얼 / 공식 예제를 따른 TabPFN fine-tuning 루프
        (classifier 버전)
        """
        optimizer = AdamW(clf.model_.parameters(), lr=self.finetune_lr)

        for epoch in range(self.finetune_epochs):
            for data_batch in tqdm(data_loader, desc=f"[TabPFN FT] epoch {epoch+1}/{self.finetune_epochs}"):
                optimizer.zero_grad()

                (
                    X_trains_preprocessed,
                    X_tests_preprocessed,
                    y_trains_preprocessed,
                    y_test_standardized,
                    cat_ixs,
                    confs,
                    normalized_bardist_,
                    bardist_,
                    batch_x_test_raw,
                    batch_y_test_raw,
                ) = data_batch

                # bar distribution (criterion) 세팅
                clf.normalized_bardist_ = normalized_bardist_[0]

                # preprocessed tensor들로 모델 초기화
                clf.fit_from_preprocessed(
                    X_trains_preprocessed,
                    y_trains_preprocessed,
                    cat_ixs,
                    confs,
                )

                # forward
                averaged_pred_logits, _, _ = clf.forward(
                    X_tests_preprocessed
                )

                # bardist_는 normalized bar distribution loss fn
                lossfn = bardist_[0]

                # per-sample NLL loss
                nll_loss_per_sample = lossfn(
                    averaged_pred_logits,
                    y_test_standardized.to(self.device),
                )

                loss = nll_loss_per_sample.mean()

                loss.backward()
                optimizer.step()

    # ---------- public: fit ----------
    def fit(self, X, y):
        """
        X: numpy array, shape (N, D)
        y: numpy array, shape (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if self.do_finetune:
            # 1) fine-tuning용 dataloader & batched classifier 준비
            clf_ft, data_loader = self._build_finetune_dataloader(X, y)

            # 2) fine-tuning 수행 (TabPFN backbone gradient update)
            self._run_finetuning(clf_ft, data_loader)

            # 3) inference-friendly 모델로 clone
            clf_eval = clone_model_for_evaluation(
                clf_ft,
                self.base_args,
                TabPFNClassifier,
            )
            # 4) clone된 모델에 정상 fit (일반 TabPFN inference에서 쓰는 preprocessing 재구성)
            clf_eval.fit(X, y)

            self.model = clf_eval
        else:
            # 기존처럼 zero-shot TabPFN 사용 (fine-tuning 없이)
            clf = TabPFNClassifier(**self.base_args)
            clf.fit(X, y)
            self.model = clf

    # ---------- public: predict_proba ----------
    def predict_proba(self, X):
        assert self.model is not None, "먼저 fit()을 호출하여 모델을 학습하세요."
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)


# ----------------------------------------------------
# 6. 전체 파이프라인
# ----------------------------------------------------
class ProductionPipeline:
    """(사전 학습된) FeatureEncoder + MainModel(TabPFN) 하이브리드 파이프라인"""

    def __init__(
        self,
        n_epochs=13,
        batch_size=32,
        n_cv_splits=5,
        encoder_weight_path="feature_encoder.pth",
        tabpfn_model_path=None,
        # fine-tune 관련 하이퍼파라미터
        do_finetune=True,
        finetune_epochs=1,
        finetune_lr=1e-5,
        max_data_size=150,
        n_estimators=2,
    ):
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
        # 오픈 TabPFN ckpt 경로
        self.tabpfn_model_path = tabpfn_model_path

        # fine-tuning 설정 저장
        self.do_finetune = do_finetune
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.max_data_size = max_data_size
        self.n_estimators = n_estimators

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

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n===== Fold {fold_idx + 1} =====")

            X_train_fold = X_train_hybrid[train_idx]
            y_train_fold = train_Y_series.values[train_idx]

            X_val = X_train_hybrid[val_idx]
            y_val = train_Y_series.values[val_idx]

            print(f"[Main] Train size: {X_train_fold.shape[0]}")
            print(f"[Main] Val size  : {X_val.shape[0]} "
                  f"(NG={ (y_val==1).sum() }, Good={ (y_val==0).sum() })")

            # TabPFN 기반 MainModel 사용 (fold마다 새로 생성)
            model = MainModel(
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_path=self.tabpfn_model_path,
                do_finetune=self.do_finetune,
                finetune_epochs=self.finetune_epochs,
                finetune_lr=self.finetune_lr,
                max_data_size=self.max_data_size,
                n_estimators=self.n_estimators,
                random_state=42 + fold_idx,  # (TabPFN 쪽에서는 무시될 수 있음)
            )
            model.fit(X_train_fold, y_train_fold)

            val_prob_ng = model.predict_proba(X_val)[:, 1]

            # 공통 평가 함수 사용 (k는 기본값 15 사용)
            roc, profit, score = calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
            )

            cv_roc_list.append(roc)
            cv_profit_list.append(profit)
            cv_score_list.append(score)

        print("\n===== CV Summary (StratifiedKFold, approx val~146) =====")
        print(f"ROC-AUC  mean/std : {np.mean(cv_roc_list):.6f} / {np.std(cv_roc_list):.6f}")
        print(f"Profit   mean/std : {np.mean(cv_profit_list):.2f} / {np.std(cv_profit_list):.2f}")
        print(f"Score    mean/std : {np.mean(cv_score_list):.6f} / {np.std(cv_score_list):.6f}")

        # 10. 제출용 모델 재학습 (Train 전체 사용 + fine-tuning)
        self.main_model = MainModel(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_path=self.tabpfn_model_path,
            do_finetune=self.do_finetune,
            finetune_epochs=self.finetune_epochs,
            finetune_lr=self.finetune_lr,
            max_data_size=self.max_data_size,
            n_estimators=self.n_estimators,
            random_state=42,
        )
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
        save_path = f"../data/submission/TabPFN_FT_submission_{timestamp}.csv"

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
        # 오픈 TabPFN ckpt 경로 지정 (필요 시)
        tabpfn_model_path=None,
        # fine-tuning 옵션
        do_finetune=True,
        finetune_epochs=1,
        finetune_lr=1e-5,
        max_data_size=150,
        n_estimators=2,
    )
    submission_result = pipeline.run_production_pipeline()

    # 간단 확인용 출력
    print("\nSubmission head:")
    print(submission_result.head())
    print("\nSubmission tail:")
    print(submission_result.tail())


if __name__ == "__main__":
    main()