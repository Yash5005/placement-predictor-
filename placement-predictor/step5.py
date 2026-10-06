import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/placement.csv")
X = df.drop(columns="placed")
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

best_name, best_model, best_acc = None, None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.3f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print("Best model:", best_name)
joblib.dump(best_model, "model.pkl")
print("Saved model.pkl")