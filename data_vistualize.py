import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

# (Optional) ensure numeric for safety
for col in ["Calories (kcal)","Carbohydrates (g)","Protein (g)","Fats (g)",
            "Free Sugar (g)","Fibre (g)","Sodium (mg)","Calcium (mg)",
            "Iron (mg)","Vitamin C (mg)","Folate (µg)"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Quick look at features
print(list(data.columns))

# -----------------------------------------------------
# 1) Distributions: Calories, Protein, Fats, Carbohydrates
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(data["Calories (kcal)"].dropna(), bins=30)
plt.title("Distribution of Calories")
plt.xlabel("Calories (kcal)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(data["Protein (g)"].dropna(), bins=30)
plt.title("Distribution of Protein")
plt.xlabel("Protein (g)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(data["Fats (g)"].dropna(), bins=30)
plt.title("Distribution of Fats")
plt.xlabel("Fats (g)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(data["Carbohydrates (g)"].dropna(), bins=30)
plt.title("Distribution of Carbohydrates")
plt.xlabel("Carbohydrates (g)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 2) Boxplot comparing Protein, Fats, Carbohydrates
# -----------------------------------------------------
plt.figure(figsize=(7, 5))
plt.boxplot(
    [data["Protein (g)"].dropna(),
     data["Fats (g)"].dropna(),
     data["Carbohydrates (g)"].dropna()],
    labels=["Protein", "Fats", "Carbohydrates"],
    showfliers=True
)
plt.title("Boxplot of Macronutrients")
plt.ylabel("grams")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 3) Correlation heatmap (with annotations + layout fixes)
# -----------------------------------------------------
corr = data.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
im = plt.imshow(corr, cmap="viridis", aspect="auto")

# ticks
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)

# annotate each cell with value
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center", color="white", fontsize=7)

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("Correlation Heatmap of Nutrients")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 4) Average micronutrients bar chart (labels not cut off)
# -----------------------------------------------------
micros = ["Calcium (mg)", "Iron (mg)", "Vitamin C (mg)", "Folate (µg)"]
means = data[micros].mean(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.bar(means.index, means.values)
plt.title("Average Micronutrient Levels")
plt.ylabel("Average Value")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 5) Missing values per feature (labels not cut off)
# -----------------------------------------------------
missing = data.isna().sum()

plt.figure(figsize=(12, 6))
plt.bar(missing.index, missing.values)
plt.title("Missing Values in Dataset")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# --- Optional: save figures instead of (or in addition to) showing ---
# Replace plt.show() above with lines like:
# plt.savefig("hist_calories.png", dpi=150, bbox_inches="tight"); plt.close()
