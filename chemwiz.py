import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Generate Synthetic Dataset
# -----------------------------
np.random.seed(42)

rows = 5000

df = pd.DataFrame({
    "Equipment_ID": [f"EQ{1000+i}" for i in range(rows)],
    "Temperature": np.random.normal(75, 10, rows),
    "Pressure": np.random.normal(40, 5, rows),
    "Flow_Rate": np.random.normal(100, 15, rows),
    "Vibration": np.random.normal(5, 1.5, rows),
    "Humidity": np.random.normal(55, 10, rows),
    "Power_Consumption": np.random.normal(500, 70, rows),
    "Operating_Hours": np.random.randint(100,5000,rows)
})

# -----------------------------
# Create Failure Label
# -----------------------------
failure = []

for i in range(rows):
    score = 0

    if df.loc[i,"Temperature"] > 85:
        score += 1

    if df.loc[i,"Pressure"] > 48:
        score += 1

    if df.loc[i,"Vibration"] > 6:
        score += 1

    if df.loc[i,"Operating_Hours"] > 4000:
        score += 1

    if score >= 2:
        failure.append(1)
    else:
        failure.append(0)

df["Failure"] = failure

# -----------------------------
# Introduce Missing Values
# -----------------------------
for col in ["Temperature","Pressure","Flow_Rate"]:
    idx = np.random.choice(df.index,50)
    df.loc[idx,col] = np.nan

# Fill Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# Remove Outliers
# -----------------------------
for col in ["Temperature","Pressure","Flow_Rate","Vibration"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1

    lower = q1-1.5*iqr
    upper = q3+1.5*iqr

    df[col] = np.clip(df[col],lower,upper)

# -----------------------------
# Feature Engineering
# -----------------------------
df["Health_Score"] = (
        100
        - (df["Temperature"]*0.3)
        - (df["Pressure"]*0.4)
        - (df["Vibration"]*2)
)

df["Health_Score"] = df["Health_Score"].clip(0,100)

df["Risk_Score"] = (
        (df["Temperature"]/100)*30 +
        (df["Pressure"]/60)*25 +
        (df["Vibration"]/10)*25 +
        (df["Operating_Hours"]/5000)*20
)

df["Risk_Score"] = df["Risk_Score"].round(2)

# -----------------------------
# Machine Learning
# -----------------------------
X = df.drop(columns=["Equipment_ID","Failure"])
y = df["Failure"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("\nAccuracy:",accuracy_score(y_test,pred))

print(classification_report(y_test,pred))

# -----------------------------
# Predict Entire Dataset
# -----------------------------
df["Prediction"] = model.predict(X)

df["Failure_Probability"] = model.predict_proba(X)[:,1]

# -----------------------------
# Save Files
# -----------------------------
df.to_csv("processed_data.csv",index=False)

prediction_df = df[[
    "Equipment_ID",
    "Prediction",
    "Failure_Probability",
    "Health_Score",
    "Risk_Score"
]]

prediction_df.to_csv("predictions.csv",index=False)

print("\nFiles Saved Successfully!")
print("processed_data.csv")
print("predictions.csv")

# -----------------------------
# KPIs
# -----------------------------
print("\n------ KPI SUMMARY ------")
print("Total Equipments :",len(df))
print("Average Temperature :",round(df["Temperature"].mean(),2))
print("Average Pressure :",round(df["Pressure"].mean(),2))
print("Average Health Score :",round(df["Health_Score"].mean(),2))
print("Failure Rate :",round(df["Prediction"].mean()*100,2),"%")