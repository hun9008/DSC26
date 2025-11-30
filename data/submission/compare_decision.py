import pandas as pd

def compare_decisions(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    merged = df1.merge(df2, on="ID", suffixes=("_1", "_2"))

    # decision 일치 여부
    merged["match"] = merged["decision_1"] == merged["decision_2"]

    total = len(merged)
    matches = merged["match"].sum()
    accuracy = matches / total * 100

    print(f"총 {total}개 중 {matches}개 일치")
    print(f"일치율: {accuracy:.2f}%")

    # probability MSE 계산
    if "probability_1" in merged.columns and "probability_2" in merged.columns:
        mse = ((merged["probability_1"] - merged["probability_2"]) ** 2).mean()
        print(f"probability MSE: {mse:.6f}")
    else:
        print("probability 컬럼을 찾을 수 없습니다. (probability_1 / probability_2)")

    return accuracy


if __name__ == "__main__":

    csv1 = '../submission_dummy/hybrid_submission_170.csv'
    csv2 = './full_ensemble_rank_20251130_143851.csv'

    compare_decisions(csv1, csv2)