# Loan Approval Prediction using Artificial Intelligence

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# 1. Load Dataset
# ---------------------------
data = pd.read_csv("loan_dataset.csv")

# ---------------------------
# 2. Data Preprocessing
# ---------------------------

# Drop Loan_ID (not useful for prediction)
data = data.drop("Loan_ID", axis=1)

# Convert categorical columns into numeric
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# ---------------------------
# 3. Split Features and Target
# ---------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Train AI Model
# ---------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 6. Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 7. Model Evaluation
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 8. Feature Importance
# ---------------------------
importances = model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance - Loan Approval Prediction")
plt.gca().invert_yaxis()
plt.show()
