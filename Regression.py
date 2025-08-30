import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

df = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

numeric_df = df.select_dtypes(include=["number"])

X = numeric_df.drop(columns=["Calories (kcal)"])
y = numeric_df["Calories (kcal)"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state= 1)

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)

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