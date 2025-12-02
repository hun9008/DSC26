import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# 1) CSV 목록 불러오기
# -----------------------------------
csv_files = sorted(glob.glob("./*.csv"))

# 특별 CSV
special_csv_path = "../submission_dummy/hybrid_submission_170.csv"
df_special = pd.read_csv(special_csv_path)
x_special = range(len(df_special))
y_special = df_special["probability"].values

n = len(csv_files)
print(f"총 CSV 개수: {n}")

# -----------------------------------
# 2) subplot 그리드 계산 (원하면 변경 가능)
# -----------------------------------
cols = 5    # 한 줄에 5개씩
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 3, rows * 2.5))

# -----------------------------------
# 3) 각 CSV를 개별 subplot에 그리기 (special CSV 겹치기)
# -----------------------------------
for i, path in enumerate(csv_files, start=1):
    df = pd.read_csv(path)
    x = range(len(df))
    y = df["probability"].values

    plt.subplot(rows, cols, i)

    # 해당 CSV 플롯 (파란색)
    plt.plot(x, y, color="blue", linewidth=1)

    # special CSV overlay (빨간색, 얇고 투명도 설정)
    plt.plot(x_special, y_special, color="red", linewidth=0.6, alpha=0.6)

    plt.title(os.path.basename(path), fontsize=7)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("submission_probability_comparison.png", dpi=300)