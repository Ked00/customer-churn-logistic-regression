import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

data = "churn.csv"
model = pd.read_csv(data)
df = model.dropna(axis=0, subset=["Tenure"])

num_features = ["Balance", "Gender"]
cat_features = ["CustomerId", "CreditScore", "Tenure", "Age", "NumOfProducts",
                "EstimatedSalary"]

X = df[num_features + cat_features]
y = df["IsActiveMember"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.70, shuffle=True)

num_pipeline = Pipeline(steps=[
    ("num", preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ("impute", SimpleImputer(strategy="mean", missing_values=np.nan))
])

cat_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="mean", missing_values=np.nan))
])

preprocesser = ColumnTransformer(transformers=[
    ("cat_features", cat_pipeline, cat_features),
    ("num_features", num_pipeline, num_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocesser),
    ("model", LogisticRegression(random_state=1))
])

mae = -1 * cross_val_score(pipeline,X, y, cv=5,scoring="neg_mean_absolute_error")
print(mae.mean())