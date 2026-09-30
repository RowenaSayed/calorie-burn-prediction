"""
Trains the Calories Burn model and saves it for the web app.

The code below is copied from Encoding_project.ipynb (same steps, same order,
same parameters, same variable names). The only additions are:
  - saving the fitted encoder, scaler and model with joblib
  - saving small metadata (input ranges, test MAE / R2) for the UI

Run once:  python train_model.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error

# Same path as the notebook. Set CALORIES_DATA to use a different location.
DATA_PATH = os.environ.get('CALORIES_DATA', r'E:\datasets\calories_Dataset.csv')

# ---------------- Load ----------------
df=pd.read_csv(DATA_PATH)
df.drop(columns=['User_ID'],inplace=True)
num_features=df.select_dtypes(include='number').columns.tolist()

# ---------------- Data cleaning ----------------
df['Weight']=df['Weight'].fillna(df['Weight'].median())
df['Heart_Rate']=df['Heart_Rate'].fillna(df['Heart_Rate'].median())
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])

df=df[~df.duplicated()]

num_cols=df[num_features]
q1=num_cols.quantile(.25)
q3=num_cols.quantile(.75)
iqr=q3-q1
lb=q1-(3*iqr)
ub=q3+(3*iqr)

for col in num_features:
    df.loc[(df[col]>ub[col])|(df[col]<lb[col]),col]=np.nan

df[num_features] = df[num_features].fillna(df[num_features].median())

# ---------------- Encoding ----------------
df = df.reset_index(drop=True)
encoder=OneHotEncoder(sparse_output=False)
gender_encoded=encoder.fit_transform(df[['Gender']])
gender_df=pd.DataFrame(gender_encoded,columns=['Gender_female', 'Gender_male'])
df = pd.concat([df.drop(columns='Gender'),gender_df],axis=1)

# ---------------- Split and scaling ----------------
x=df.drop(columns=['Calories'])
y=df['Calories']
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# ---------------- Model (Lasso: best model in the report) ----------------
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(x_train, y_train)
y_pred= lasso_model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("R2:", r2)
print("MAE:", mae)

# ---------------- Save for the web app ----------------
os.makedirs('model', exist_ok=True)
joblib.dump({'encoder': encoder, 'scaler': scaler, 'model': lasso_model}, 'model/calories_model.joblib')

# Allowed input ranges = min/max of the cleaned training data
input_features = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
metadata = {
    'ranges': {c: [float(x[c].min()), float(x[c].max())] for c in input_features},
    'genders': encoder.categories_[0].tolist(),
    'mae': float(mae),
    'r2': float(r2),
}
with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved model/calories_model.joblib and model/metadata.json")
