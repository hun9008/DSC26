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

## baseline_140

| file | details | ROC-AUC | Net Profit | Total Score | Submission Score |
| --- | --- | --- | --- | --- | --- |
| sample_code_eval.py | basecode | 0.763127 | 220 | 0.111060 | 0.34052 |
| RF_main.py | CNN + RandomForest | 0.854064 | 2320 | 0.283572 | 0.41330 |
| RF_main_k_fold_140_naive_encoder | MLP + RandomForest | 0.729769 | -1460 | 0.046882 | 0.24494 |
| ensemble_main_kfold_140 | CNN + Ensemble(RF,ET,GB,HGB,SVM) voting | 0.868372 | 2740 | 0.312816 | X |
| ensemble_main_kfold_140_optuna | CNN + Ensemble(RF,ET,GB,HGB,SVM) voting + optuna | 0.876271 | 3580 | 0.364064 | 0.22784 |
| ensemble_main_HC_6 | CNN + Ensemble 6 Hill Clibing | 0.884339 | 3580 | 0.367987 | X |
| ensemble_main_HC_13 | CNN + Ensemble 13 Hill Clibing | 0.887789 | 3160 | 0.346466 | X |
| ensemble_main_HC_top6 | CNN + Ensemble top 6 Hill Clibing | 0.888345 | 3160 | 0.316504 | X |
| ensemble_main_HC_top10 | CNN + Ensemble top 10 Hill Clibing | 0.888468 | 3160 | 0.346875 | X |
| ensemble_main_HC_top15 | CNN + Ensemble top 15 Hill Clibing | 0.887410 | 3160 | 0.346315 | X |
