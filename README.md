# 제 5회 KAIST-POSTECH-UNIST AI & 데이터사이언스 경진대회

## 폴더 구조

```
DSC26/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── submission/
│   │   ├── CNN_RF_submission.csv
│   │   └── sample_submission.csv
│   └── submission_dummy/
├── src/
│   ├── CNN_encoder.py
│   ├── RF_main.py
│   ├── sample_code_eval.py
│   ├── sample_code.py
│   └── util/
│       └── eval.py
├── weight/
│   └── feature_encoder.pth
├── dummy/
│   ├── CNN_MLP_Ensemble_cross_val_optuna_v2.py
│   ├── CNN_MLP_Ensemble_cross_val_optuna.py
│   ├── CNN_MLP_Ensemble_cross_val.py
│   ├── CNN_MLP_Ensemble.py
│   ├── CNN_MLP_RandomForest_cross_val_v2_same_eval.py
│   ├── CNN_MLP_RandomForest_cross_val_v2.py
│   ├── CNN_MLP_RandomForest_cross_val.py
│   ├── CNN_MLP_RandomForest_v3.py
│   ├── CNN_MLP_RandomForest_val.py
│   ├── CNN_MLP_RandomForest.py
│   ├── E2E 2DCNN+RandomForest.py
│   ├── E2E ViT+RandomForest.py
│   ├── eval_form.py
│   ├── evaluation_form.py
│   ├── feature_ablation.ipynb
│   ├── naive_ensemble.py
│   ├── raw_E2E.py
│   ├── RF.py
│   ├── sample_code_cross_val.py
│   ├── sample_code.ipynb
│   ├── sample_code.py
│   ├── csv_modify.py
│   ├── best_model.pth
│   ├── feature_encoder.pth
│   └── production_model.pth
├── catboost_info/
└── README.md
```

### 데이터

학습데이터는 data/train.csv, data/test.csv 사용.
제출 결과물은 submission/ 아래 저장.
이전 제출 버전은 submission_dummy/* 에 존재.

### source code

CNN encoder를 별도로 학습. 

```
cd src
python3 CNN_encoder.py
```

이후 _main.py 모델 코드 실행으로 submission 생성.

```
cd src
python3 RF_main.py
```

### evaluation code (./src/util/eval.py)

평가부분은 따로 분리하였습니다. 
./src/util/eval.py 에 define 되어 있고 _main.py 에서 import 하여 사용합니다. 

```
# RF_main.py 참고.
from util.eval import (
    evaluate_score_general,
    calculate_competition_score,
)

# 대회 조건 평가
calculate_competition_score(
                y_true=y_val,
                y_prob=val_prob_ng,
                k=15,
                profit_good=100,
                cost_ng=2000
            )
```

### logger code (./src/util/logger.py)

로그 모듈도 따로 분리하였습니다. 
./src/util/logger.py 에 define 되어 있고 _main.py, _encoder.py 에서 import 하여 사용합니다. 

```
# RF_main.py 참고.
from util.logger import TeeLogger
import sys

# 학습 혹은 로그 남길 부분에 아래 코드 삽입

logger = TeeLogger()
sys.stdout = logger

### 실행 코드 ###

logger.close()
sys.stdout = sys.__stdout__
print(f"[Main] Log saved to: {logger.log_path}")
```

## eval_v1 leader board

| file | details | ROC-AUC | Net Profit | Total Score | Submission Score |
| --- | --- | --- | --- | --- | --- |
| sample_code_eval.py | basecode | 0.763127 | 220 | 0.111060 | 0.34052 |
| RF_main_kfold_140.py | CNN + RandomForest | 0.854064 | 2320 | 0.283572 | 0.41330 |
| RF_main_k_fold_140_naive_encoder | MLP + RandomForest | 0.729769 | -1460 | 0.046882 | 0.24494 |
| ensemble_main_kfold_140 | CNN + Ensemble(RF,ET,GB,HGB,SVM) voting | 0.868372 | 2740 | 0.312816 | X |
| ensemble_main_kfold_140_optuna | CNN + Ensemble(RF,ET,GB,HGB,SVM) voting + optuna | 0.876271 | 3580 | 0.364064 | 0.22784 |
| RF_main_kfold_140_tiny_CNN_encoder_param745.py | Tiny CNN(param : 745) | 0.805122 | 1480 | 0.166432 | 0.28737 |
| RF_main_kfold_140_tiny_CNN_encoder_param39657.py | Tiny CNN(param : 39657) | 0.881620 | 2320 | 0.238705 | 0.08948 |
| RF_main_kfold_140_1dCNN_encoder.py | 1d CNN | 0.867560 | 3160 | 0.337542 | 0.37968 |
| ensemble_main_HC_6 | CNN + Ensemble 6 Hill Clibing | 0.884339 | 3580 | 0.367987 | X |
| ensemble_main_HC_13 | CNN + Ensemble 13 Hill Clibing | 0.887789 | 3160 | 0.346466 | X |
| ensemble_main_HC_top6 | CNN + Ensemble top 6 Hill Clibing | 0.888345 | 3160 | 0.316504 | X |
| ensemble_main_HC_top10 | CNN + Ensemble top 10 Hill Clibing | 0.888468 | 3160 | 0.346875 | X |
| ensemble_main_HC_top15 | CNN + Ensemble top 15 Hill Clibing | 0.887410 | 3160 | 0.346315 | X |
| ensemble_main_HC_top20 | CNN + Ensemble top 20 Hill Clibing | 0.888403 | 3160 | 0.316392 | X |
| deep ensemble_main | deep ensemble (MLPs) | 0.881606 | -2700 | 0.000000 | 0.25625 |
| RF_main_kfold_140_AE | RF + AE | 0.856430 | 3160 | 0.300320 | 0.21593 |
| RF_main_kfold_140_GCN | RF + GCN | 0.905749 | 4000 | 0.402694 | 0.00000 |
| SVM_main_kfold_140 | SVM + GCN | 0.884222 | 2740 | 0.290704 | 0.00000 |
| RF_main_kfold_140_GCN_CNN | RF + GCN + CNN | 0.888259 | 3160 | 0.345009 | 0.00000 |
| full_ensemble_rank | fraud dectection style | 0.760973 | 4000 | 0.323093 | 0.25163 |



## eval_v2 leader board

| file | details | ROC-AUC | Net Profit | Total Score | Submission Score |
| --- | --- | --- | --- | --- | --- |
| sample_code_eval_v2.py | basecode | 0.936561 | 20000 | 0.934410 | 0.34052 |
| RF_main_kfold_140_v2.py | CNN + RandomForest | 0.854408 | 15800 | 0.748308 | 0.41330 |

## compare check

hybrid_submission_170 (0.52622) 버전과의 유사도 비교
accuracy (TF 일치여부) 가 78% 미만에서는 0.4 이상의 submission score 없음.
참고용으로 사용가능할듯

| id | file |   accuracy | probability_MSE  | submission score |
| --- | --- | --- | --- | --- |
|20 |                         hybrid_submission_150.csv  | 95.708155   |      0.000000 | 0.49099
|21 |                    hybrid_submission_reupload.csv  | 93.562232   |      0.000000 | 0.50084
|16 |    ensemble_submission_10runs_20251125_145558.csv  | 81.115880   |      0.009833 | 0.23941
|6  |                             CNN_RF_submission.csv  | 80.257511   |      0.018251 | 0.41330
|4  |            CNN_MLP_RF_CV_withVal60_submission.csv  | 78.540773   |      0.016994 | 0.22090
|14 |    ensemble_optuna_submission_20251127_143822.csv  | 78.540773   |      0.018176 | 0.34231
|1  | CNN_Extractor_RF_submission_20251129_032125_18...  | 78.111588   |      0.021018 | 0.45238
|0  |          CNN_AE_RF_submission_20251129_172303.csv  | 77.682403   |      0.018185 | 0.21593
|3  |    CNN_MLP_EnsembleVoting_OptunaCV_submission.csv  | 77.682403   |      0.015645 | 0.34231
|2  |         CNN_GCN_RF_submission_20251130_133302.csv  | 76.824034   |      0.022058 | 0.00000
|19 |            full_ensemble_rank_20251130_143851.csv  | 76.824034   |      0.162137 | 0.25163
|23 |               hybrid_with_ablation_submission.csv  | 76.394850   |      0.013693 | 0.32606
|26 |               submission_randomforest_display.csv  | 75.965665   |      0.019291 | 0.30644
|11 | Tiny_CNN_RF_kfold_140_submission_20251128_1551...  | 75.536481   |      0.023223 | 0.28737
|15 |                           ensemble_submission.csv  | 75.536481   |      0.026203 | 0.00000
|17 |                 first_ablation_cnn_submission.csv  | 75.536481   |      0.026269 | 0.00000
|9  |            DeepEns_Focal_OOFk_20251129_154907.csv  | 75.536481   |      0.078708 | 0.25652
|8  |            CNN_SVM_submission_20251130_130055.csv  | 75.536481   |      0.021832 | 0.00000
|5  | CNN_RF_kfold_140_naive_encoder_submission_2025...  | 75.107296   |      0.028884 | 0.24494
|18 |                     first_ablation_submission.csv  | 74.248927   |      0.028029 | 0.24012
|24 |                            my_submission_test.csv  | 74.248927   |      0.025077 | 0.32112
|13 |                      convnext_sota_submission.csv  | 74.248927   |      0.026795 | 0.33081
|12 | Tiny_CNN_RF_kfold_140_submission_20251128_2254...  | 73.819742   |      0.026762 | 0.08948
|7  |                       CNN_RF_submission_NoPad.csv  | 72.961373   |      0.030370 | 0.36073
|22 |                         hybrid_vit_submission.csv  | 71.673820   |      0.027752 | 0.18614
|10 |             GCN_RF_submission_20251129_215257.csv  | 71.244635   |      0.034964 | 0.00000
|25 |       submission_efficientnet_b0_randomforest.csv  | 70.386266   |      0.032399 | 0.34951
