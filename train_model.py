import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


print("🚀 Script Started...\n")


# ================= DATA CLEANING FUNCTION =================
def clean_data(df):
    print("🧹 Cleaning data...")

    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    print("✅ Data cleaned\n")
    return df


# ================= TRAIN MODELS FUNCTION =================
def train_models(X_train, X_test, y_train, y_test, name="model"):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    best_model = None
    best_acc = 0
    best_cm = None
    accuracies = {}

    print(f"\n📊 Training models for {name}...\n")

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            report = classification_report(y_test, preds)

            accuracies[model_name] = acc

            print(f"🔹 {model_name} Accuracy: {acc:.4f}")
            print(f"{model_name} Confusion Matrix:\n{cm}\n")
            print(f"{model_name} Classification Report:\n{report}\n")

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_cm = cm

        except Exception as e:
            print(f"❌ Error in {model_name}: {e}")

    print(f"🏆 Best Model: {type(best_model).__name__} ({best_acc:.4f})\n")

    # ================= CONFUSION MATRIX PLOT =================
    if best_cm is not None:
        plt.figure()
        plt.imshow(best_cm, cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for i in range(len(best_cm)):
            for j in range(len(best_cm)):
                plt.text(j, i, best_cm[i][j], ha="center", va="center", color="black")

        cm_file = f"{name}_confusion_matrix.png"
        plt.savefig(cm_file)
        plt.close()
        print(f"📊 Saved: {cm_file}")

    # ================= ACCURACY GRAPH =================
    plt.figure()
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title(f"{name} Model Accuracy Comparison")
    plt.xticks(rotation=30)

    acc_file = f"{name}_accuracy_comparison.png"
    plt.savefig(acc_file)
    plt.close()
    print(f"📈 Saved: {acc_file}\n")

    return best_model


# ================= HEART =================
print("🔄 Training Heart Disease Model...")

if not os.path.exists("heart.csv"):
    print("❌ ERROR: heart.csv not found!")
else:
    heart_df = pd.read_csv("heart.csv")
    heart_df.columns = heart_df.columns.str.strip()

    heart_df.rename(columns={
        "MaxHR": "thalach",
        "max_hr": "thalach",
        "thalachh": "thalach",
        "BloodPressure": "trestbps",
        "Cholesterol": "chol"
    }, inplace=True)

    heart_df = clean_data(heart_df)

    required_cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal","target"]

    missing = [col for col in required_cols if col not in heart_df.columns]

    if missing:
        print(f"❌ Missing columns: {missing}")
    else:
        X = heart_df.drop("target", axis=1)
        y = heart_df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = train_models(X_train, X_test, y_train, y_test, "heart")

        if model:
            pickle.dump(model, open("heart_model.pkl", "wb"))
            pickle.dump(scaler, open("heart_scaler.pkl", "wb"))
            pickle.dump(list(X.columns), open("heart_features.pkl", "wb"))
            print("✅ ❤️ Heart Model Saved\n")


# ================= DIABETES =================
print("🔄 Training Diabetes Model...")

file_name = None
for f in os.listdir():
    if ("diabetes" in f.lower()) and f.endswith(".csv"):
        file_name = f
        break

if file_name is None:
    print("❌ No diabetes dataset found!")
else:
    print(f"✅ Using file: {file_name}")

    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()

    df = clean_data(df)

    if "Outcome" not in df.columns:
        print("❌ 'Outcome' column missing!")
    else:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = train_models(X_train, X_test, y_train, y_test, "diabetes")

        if model:
            pickle.dump(model, open("diabetes_model.pkl", "wb"))
            pickle.dump(scaler, open("diabetes_scaler.pkl", "wb"))
            pickle.dump(list(X.columns), open("diabetes_features.pkl", "wb"))
            print("✅ 🩸 Diabetes Model Saved\n")


print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")