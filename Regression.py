import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

numeric_df = df.select_dtypes(include=["number"])

# Split data into features and target
X = numeric_df.drop(columns=["Calories (kcal)"])
y = numeric_df["Calories (kcal)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state= 1)

# use pipeline to handle missing values and model training
pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)

# Make predictions
y_pred = pipe.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

sample_vals = {'Carbohydrates (g)': 10,
               'Protein (g)': 10,
               'Fats (g)': 10,
               'Free Sugar (g)': 10,
               'Fibre (g)': 10,
               'Sodium (mg)': 10,
               'Calcium (mg)': 10,
               'Iron (mg)': 110,
               'Vitamin C (mg)': 110,
               'Folate (µg)': 110 }

sample = pd.DataFrame([sample_vals], columns=X.columns)
predicted_calories = pipe.predict(sample)
print("Predicted Calories :", predicted_calories[0])

df['Calories Difference'] = abs(df['Calories (kcal)'] - predicted_calories[0])
closest_dishes = df.nsmallest(5, 'Calories Difference')
print("5 Closest Dishes:")
print(closest_dishes[['Dish Name', 'Calories (kcal)', 'Calories Difference']])

# plot regression diagnostics

def plot_regression_diagnostics(y_true, y_pred):
    """Save 3 standard regression visuals: Pred vs Actual, Residuals vs Pred, Residuals Hist."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12,c='red')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, s=12)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Residuals histogram
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.show()
    plt.close()

plot_regression_diagnostics(y_test, y_pred)