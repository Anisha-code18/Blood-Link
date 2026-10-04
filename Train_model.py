import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
 
# ─── Step 1: Load Feature Engineered Dataset ──────────────────────
 
pairs_df = pd.read_csv("feature_engineered.csv")
 
# ─── Step 2: Separate X and y ─────────────────────────────────────
 
X = pairs_df[["blood_match", "distance_km",
              "days_since_donation", "urgency_score",
              "is_eligible"]]
 
y = pairs_df["compatibility"]
 
# ─── Step 3: Split into Train and Test ────────────────────────────
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
print("=" * 55)
print("  DATASET SPLIT")
print("=" * 55)
print(f"Total rows      : {len(pairs_df)}")
print(f"Training rows   : {len(X_train)}")
print(f"Testing rows    : {len(X_test)}")
 
# ─── Step 4: Train the Random Forest Model ────────────────────────
 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n" + "=" * 55)
print("  MODEL TRAINED SUCCESSFULLY ✅")
print("=" * 55)
 
# ─── Step 5: Evaluate the Model ───────────────────────────────────
 
y_pred = model.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
 
print(f"\nAccuracy Score  : {round(accuracy * 100, 2)}%")
 
print("\n" + "=" * 55)
print("  CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_test, y_pred,
      target_names=["Not Compatible", "Compatible"]))
 
# ─── Step 6: Feature Importance ───────────────────────────────────
 
print("=" * 55)
print("  FEATURE IMPORTANCE")
print("=" * 55)
importances = model.feature_importances_
feature_names = X.columns
for name, score in sorted(zip(feature_names, importances),
                           key=lambda x: x[1], reverse=True):
    print(f"{name:<25} : {round(score * 100, 2)}%")
 
# ─── Step 7: Save the Model ───────────────────────────────────────
 
with open("blood_match_model.pkl", "wb") as f:
    pickle.dump(model, f)
 
print("\nModel saved as blood_match_model.pkl ✅")