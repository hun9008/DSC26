import pandas as pd
import glob
import os

def compare_decisions_single(df1, df2, name="(unknown)"):
    merged = df1.merge(df2, on="ID", suffixes=("_1", "_2"))

    # decision 일치 여부
    merged["match"] = merged["decision_1"] == merged["decision_2"]

    total = len(merged)
    matches = merged["match"].sum()
    acc = matches / total * 100

    # probability MSE
    if "probability_1" in merged.columns and "probability_2" in merged.columns:
        mse = ((merged["probability_1"] - merged["probability_2"]) ** 2).mean()
    else:
        mse = None

    print(f"\n=== {name} ===")
    print(f"총 {total}개 중 {matches}개 일치  (일치율 {acc:.2f}%)")
    if mse is not None:
        print(f"probability MSE: {mse:.6f}")
    else:
        print("probability_1 / probability_2 컬럼 없음")

    return acc, mse


def main():
    csv1_path = '../submission_dummy/hybrid_submission_170.csv'
    folder_path = '../submission_test/*.csv'

    df1 = pd.read_csv(csv1_path)

    results = []

    for csv2_path in sorted(glob.glob(folder_path)):
        name = os.path.basename(csv2_path)
        df2 = pd.read_csv(csv2_path)

        acc, mse = compare_decisions_single(df1, df2, name=name)

        results.append({
            "file": name,
            "accuracy": acc,
            "probability_MSE": mse
        })

    # 결과 정리 DataFrame 출력
    print("\n\n================ SUMMARY ================")
    summary_df = pd.DataFrame(results)
    print(summary_df.sort_values("accuracy", ascending=False))


if __name__ == "__main__":
    main()