# Importing the libarrires
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
## Load the data
customer_features = pd.read_csv(
    "data/customer_features.csv"
)
# target variable
customer_features["retained"] = (
    customer_features["total_orders"] > 1
).astype(int)
X = customer_features[
    [
        "recency_days",
        "total_spent_log",
        "avg_order_value"
    ]
]

y = customer_features["retained"]
# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Train the model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
# Evaluate the model
y_pred = rf.predict(X_test)

print(classification_report(
    y_test,
    y_pred
))
# make directory for saving the model
Path("models").mkdir(exist_ok=True)
# feature importance
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
})

print(feature_importance.sort_values(
    by="importance",
    ascending=False
))
# saving it as a csv file
feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

# Save the model
joblib.dump(
    rf,
    "models/retention_model.pkl"
)

print("Model saved successfully.")
# visualize feature importance
plt.figure(figsize=(8,5))

plt.bar(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("models/feature_importance.png")
plt.close()