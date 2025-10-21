import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

numeric_df = df.select_dtypes(include=["number"])

# Clean data (Drop row that have missing value)
cleaned_df = numeric_df.dropna()

# Split data into features and target
X = cleaned_df.drop(columns=["Calories (kcal)","Free Sugar (g)","Fibre (g)","Sodium (mg)","Calcium (mg)","Iron (mg)","Vitamin C (mg)","Folate (µg)"])
y = cleaned_df["Calories (kcal)"]

#Testing model
random_seed = [1,2,3,5,7,99,100,666,2025,9999]

for seed in random_seed:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    pipeline = make_pipeline(
        StandardScaler(),
        LinearRegression(),)
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Print in CSV format
    print(f"{seed},{mse:.3f},{mae:.3f},{r2:.3f},{rmse:.3f}")
