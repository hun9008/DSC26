import pandas as pd

def compare_decisions(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    merged = df1.merge(df2, on="ID", suffixes=("_1", "_2"))

    merged["match"] = merged["decision_1"] == merged["decision_2"]

    total = len(merged)
    matches = merged["match"].sum()
    accuracy = matches / total * 100

    print(f"총 {total}개 중 {matches}개 일치")
    print(f"일치율: {accuracy:.2f}%")


    return accuracy


if __name__ == "__main__":

    # import sys
    # if len(sys.argv) != 3:
    #     print("Usage: python compare_decision.py <csv1> <csv2>")
    #     exit(1)

    csv1 = '../submission_dummy/hybrid_submission_170.csv'
    csv2 = './GCN_RF_submission_20251129_215257.csv'

    compare_decisions(csv1, csv2)