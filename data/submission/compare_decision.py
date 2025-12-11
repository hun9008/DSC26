import pandas as pd

def compare_decisions(csv1_path, csv2_path, score):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    merged = df1.merge(df2, on="ID", suffixes=("_1", "_2"))

    # decision 일치 여부
    merged["match"] = merged["decision_1"] == merged["decision_2"]

    total = len(merged)
    matches = merged["match"].sum()
    accuracy = matches / total * 100

    # print(f"총 {total}개 중 {matches}개 일치")
    print(f"일치율: {accuracy:.2f}%\t\t{score}")

    # probability MSE 계산
    # if "probability_1" in merged.columns and "probability_2" in merged.columns:
    #     # mse = ((merged["probability_1"] - merged["probability_2"]) ** 2).mean()
    #     # print(f"probability MSE: {mse:.6f}")
    # else:
    #     print("probability 컬럼을 찾을 수 없습니다. (probability_1 / probability_2)")

    return accuracy

def summary(csvs, scores, target):

    accs = []
    for csv, score in zip(csvs, scores):
        accs.append(compare_decisions(csv, target, score))
        print()
    print("avg acc : ", sum(accs)/len(accs))

if __name__ == "__main__":

    csv1 = '../submission_dummy/hybrid_submission_170.csv' #0.52622
    csv2 = '../submission_dummy/CNN_Extractor_RF_submission_20251202_223004.csv' #0.50786
    # csv3 = '../submission_test/CNN_3232_nest_200_RS_999.csv' #0.43959
    csv3 = './CNN_3232_NEST_200_RS_1_20251208_174937.csv' #0.53087
    csv4 = './150_CNN_3232_NEST_200_RS_1_20251208_175242.csv' #0.53746
    csv5 = './CNN_3232_NEST_200_RS_19_20251209_195621.csv' # 0.52655
    csv6 = './CNN_3232_NEST_200_RS_22_20251210_133118.csv' # 0.51678

    scores = [0.52622, 0.50786, 0.53087, 0.53746, 0.52655, 0.51678]

    # csv4 = './CNN_MLP_KL_submission_20251206_220512.csv'
    # target = './CNN_MLP_KL_submission_20251206_220512.csv'
    # target = 'CNN_3232_NEST_200_RS_510_20251207_150813.csv'
    target = './CNN_3232_NEST_200_RS_10_20251208_174958.csv'

    csvs = [csv1, csv2, csv3, csv4, csv5, csv6]

    # compare_decisions(csv1, target, 0.52622)
    # compare_decisions(csv2, target, 0.50786)
    # compare_decisions(csv3, target, 0.53087)
    # compare_decisions(csv4, target, 0.53746)

    # print()

    # target = csv1
    # # compare_decisions(csv1, target, 0.52622)
    # compare_decisions(csv2, target, 0.50786)
    # compare_decisions(csv3, target, 0.53087)
    # compare_decisions(csv4, target, 0.53746)
    # print()

    # target = csv2
    # compare_decisions(csv1, target, 0.52622)
    # # compare_decisions(csv2, target, 0.50786)
    # compare_decisions(csv3, target, 0.53087)
    # compare_decisions(csv4, target, 0.53746)
    # print()

    # target = csv3
    # compare_decisions(csv1, target, 0.52622)
    # compare_decisions(csv2, target, 0.50786)
    # # compare_decisions(csv3, target, 0.53087)
    # compare_decisions(csv4, target, 0.53746)
    # print()

    # target = csv4
    # compare_decisions(csv1, target, 0.52622)
    # compare_decisions(csv2, target, 0.50786)
    # compare_decisions(csv3, target, 0.53087)
    # compare_decisions(csv4, target, 0.53746)

    target = './CNN_3232_NEST_200_RS_510_20251207_150813.csv'
    summary(csvs, scores, target)
    print()

    target = './CNN_3232_NEST_200_RS_10_20251208_174958.csv'
    summary(csvs, scores, target)
    print()

    target = './CNN_3232_NEST_200_RS_354_20251207_162131.csv'
    summary(csvs, scores, target)
    print()

    target = './CNN_3232_NEST_200_RS_18_20251209_195609.csv'
    summary(csvs, scores, target)
    print()

    target = './DeepEns_Focal_OOFk_20251202_213252.csv'
    summary(csvs, scores, target)
    print()

    target = './ensemble_optuna_submission_20251127_143822.csv'
    summary(csvs, scores, target)
    print()

    target = './TabPFN_lightCNN_20251204_001917.csv'
    summary(csvs, scores, target)
    print()

    target = '../submission_dummy/CNN_Extractor_RF_submission_20251129_032125_180.csv'
    summary(csvs, scores, target)
    print()