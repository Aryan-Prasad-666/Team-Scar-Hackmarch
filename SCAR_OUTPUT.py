import pandas as pd
import numpy as np
import joblib

try:
    model_bundle = joblib.load("C:\\Users\\Aryan Prasad\\Desktop\\HACKMARCH\\KLEHM032.pkl")
    model = model_bundle["model"]
    target_encoder = model_bundle["target_encoder"]
    label_encoders = model_bundle["label_encoders"]
    feature_columns = model_bundle["feature_columns"]
    print("Model loaded successfully from KLEHM032.pkl")
    print("Required feature columns:", feature_columns)
except FileNotFoundError as e:
    print(f"Error: Missing model file - {e}")
    exit()
except Exception as e:
    print(f"Error loading model file: {e}")
    exit()

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

form_data = {}
for col in feature_columns:
    while True:
        if col == "Age":
            value = input(f"Enter {col} (numeric, 0-120): ")
            processed_value = clean_age(value)
        elif col == "Academic Performance (CGPA/Percentage)":
            value = input(f"Enter {col} (numeric, e.g., 0-10 or 0-100): ")
            processed_value = cgpa_academic(value)
        elif col == "Financial Stability - self/family (1 is low income and 10 is high income)":
            value = input(f"Enter {col} (numeric, 1-10): ")
            processed_value = clean_numeric(value)
        else: 
            valid_values = set(label_encoders.get(col, {}).classes_) if col in label_encoders else set()
            value = input(f"Enter {col} (e.g., {', '.join(list(valid_values)[:3]) if valid_values else 'any text'}): ").strip()
            if not value or (valid_values and value not in valid_values):
                value = "None"
            processed_value = value

        if col in ["Age", "Academic Performance (CGPA/Percentage)", "Financial Stability - self/family (1 is low income and 10 is high income)"]:
            if not np.isnan(processed_value):
                form_data[col] = [processed_value]
                break
            print("Invalid input. Please enter a valid number.")
        else:
            form_data[col] = [processed_value]
            break

new_data = pd.DataFrame(form_data)

new_data["Age"] = new_data["Age"].fillna(19.0)
new_data["Academic Performance (CGPA/Percentage)"] = new_data["Academic Performance (CGPA/Percentage)"].fillna(7.5)  
new_data["Financial Stability - self/family (1 is low income and 10 is high income)"] = new_data["Financial Stability - self/family (1 is low income and 10 is high income)"].fillna(5.0)  # Placeholder median

categorical_cols = [col for col in feature_columns if col not in ["Age", "Academic Performance (CGPA/Percentage)", "Financial Stability - self/family (1 is low income and 10 is high income)"]]
for col in categorical_cols:
    if col in label_encoders:
        le = label_encoders[col]
        valid_values = set(le.classes_)
        none_index = np.where(le.classes_ == "None")[0][0] if "None" in le.classes_ else 0
        new_data[col] = new_data[col].fillna("None").apply(
            lambda x: x if x in valid_values else "None")
        new_data[col] = new_data[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in valid_values else none_index)
    else:
        new_data[col] = new_data[col].fillna("None").astype(str)

new_data = new_data[feature_columns]

try:
    prediction = model.predict(new_data)
    predicted_label = target_encoder.inverse_transform(prediction)[0]
    print(f"\nPredicted Career: {predicted_label}")
except Exception as e:
    print(f"Prediction error: {str(e)}")