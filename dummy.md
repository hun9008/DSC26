
### dummy (ignored)

| file | details | ROC-AUC | Total | Submission Score |
| --- | --- | --- | --- | --- |
| sample_code_cross_val.py | base code | 0.770370 | 0.447200 | 0.34052 |
| CNN_MLP_RandomForest_cross_val.py | Feature Encoder : CNN + MLP / Main Model : RandomForest | 0.857778 | 0.770516 | 0.52622 |
| E2E ViT+RandomForest.py | Feature Encoder : ViT + MLP / Main Model : RandomForest | 1.0 | 1.0 | 0.18614 | 
| CNN_MLP_Ensemble.py | Feature Encoder : CNN + MLP / Main Model : {RandomForest, ExtraTrees, GradientBoosting, HistGB, SVM} voting | 0.866667 | 0.523397 | Not submit | 

| file | details | ROC-AUC | Total | Submission Score |
| --- | --- | --- | --- | --- |
| CNN_MLP_RF|  |  |  | 0.22090 | 
| CNN_MLP_Ensemble_optuna.py | Feature Encoder : CNN + MLP / Main Model : {RandomForest, ExtraTrees, GradientBoosting, HistGB, SVM} voting | 0.899556 | 0.546529 | 0.34231 | 


## baseline

| file | details | ROC-AUC | Net Profit | Total Score | Submission Score |
| --- | --- | --- | --- | --- | --- |
| sample_code_eval.py | basecode | 0.937034 | 1500 | 0.256269 | 0.34052 |
| RF_main.py | CNN + RandomForest | 0.836593 | -180 | 0.138163 | 0.41330 |