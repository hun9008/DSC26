# 제 5회 KAIST-POSTECH-UNIST AI & 데이터사이언스 경진대회

## baseline

| file | details | ROC-AUC | Total | Submission Score |
| --- | --- | --- | --- | --- |
| sample_code.py | base code | 0.937034 | 0.743538 | 0.34052 |
| E2E 2DCNN+RandomForest.py | Feature Encoder : CNN + MLP / Main Model : RandomForest | 1.0 | 1.0 | 0.52622 |
| E2E ViT+RandomForest.py | Feature Encoder : ViT + MLP / Main Model : RandomForest | 1.0 | 1.0 | 0.18614 | 
| CNN_MLP_Ensemble.py | Feature Encoder : CNN + MLP / Main Model : {DT, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, HistGB, XGBoost, LightGBM, CatBoost, SVM} voting | 1.0 | 1.0 | Not submit | 

add one
