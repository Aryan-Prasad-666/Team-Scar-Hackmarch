import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

data = pd.read_csv("C:\\Users\\Aryan Prasad\\Desktop\\HACKMARCH\\train_data1.csv")

data = data.drop(columns=[
    "Timestamp", "Daily Water Intake (in Litres) ", "Favorite Color", 
    "Preferred Music Genre", "Birth Month ", "Number of Siblings"
])

def clean_age(x):
    try:
        x = str(x).strip()
        numeric_part = ''.join(c for c in x if c.isdigit() or c == '.')
        if not numeric_part:
            return np.nan
        age = float(numeric_part)
        return age if 0 < age <= 120 else np.nan  
    except (ValueError, TypeError):
        return np.nan

def cgpa_academic(x):
    try:
        x = str(x).replace("%", "").strip()
        x = float(x)
        if x <= 1: 
            return (x * 100) / 10
        elif x > 10 and x <= 100: 
            return x / 10
        return x 
    except:
        return np.nan

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

data["Age"] = data["Age"].apply(clean_age)
print("Age column after clean_age (first 50 values):", data["Age"].head(50).tolist()) 
data["Age"] = data["Age"].fillna(data["Age"].median())  

data["Academic Performance (CGPA/Percentage)"] = data["Academic Performance (CGPA/Percentage)"].apply(cgpa_academic)
data["Risk-Taking Ability "] = data["Risk-Taking Ability "].apply(clean_numeric)
data["Financial Stability - self/family (1 is low income and 10 is high income)"] = data[
    "Financial Stability - self/family (1 is low income and 10 is high income)"].apply(clean_numeric)

fill_values = {
    "Participation in Extracurricular Activities": "None",
    "Previous Work Experience (If Any)": "None",
    "Leadership Experience": "None",
    "Networking & Social Skills": "None",
    "Academic Performance (CGPA/Percentage)": data["Academic Performance (CGPA/Percentage)"].median(),
    "Risk-Taking Ability ": data["Risk-Taking Ability "].median(),
    "Financial Stability - self/family (1 is low income and 10 is high income)": data[
        "Financial Stability - self/family (1 is low income and 10 is high income)"].median()
}
data.fillna(fill_values, inplace=True)

categorical_cols = ["Gender", "Highest Education Level", "Preferred Subjects in Highschool/College",
                    "Preferred Work Environment", "Participation in Extracurricular Activities", 
                    "Previous Work Experience (If Any)", "Leadership Experience", "Networking & Social Skills", 
                    "Tech-Savviness", "Motivation for Career Choice "]

label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

target_encoder = LabelEncoder()
data["What would you like to become when you grow up"] = data["What would you like to become when you grow up"].fillna("Other")
data["What would you like to become when you grow up"] = target_encoder.fit_transform(
    data["What would you like to become when you grow up"])

drop_cols = ["Gender", "Motivation for Career Choice ", "Highest Education Level", 
             "Participation in Extracurricular Activities", "Risk-Taking Ability "]
X = data.drop(columns=["What would you like to become when you grow up"] + drop_cols)
y = data["What would you like to become when you grow up"]

categorical_cols = [col for col in categorical_cols if col in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = HistGradientBoostingClassifier(
    max_iter=300, 
    max_depth=2, 
    learning_rate=0.05, 
    random_state=42,
    l2_regularization=10.0,
    min_samples_leaf=20,
    categorical_features=[X.columns.get_loc(col) for col in categorical_cols]
)

model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f" Train Accuracy: {train_acc:.4f}")
print(f" Test Accuracy: {test_acc:.4f}")

model_bundle = {
    "model": model,
    "target_encoder": target_encoder,
    "label_encoders": label_encoders,
    "feature_columns": X.columns.tolist()
}

joblib.dump(model_bundle, "C:\\Users\\Aryan Prasad\\Desktop\\HACKMARCH\\KLEHM032.pkl")
print(" All components saved in one file: 'KLEHM032.pkl'")