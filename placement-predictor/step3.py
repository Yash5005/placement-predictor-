import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "cgpa": [6.5, 8.2, 7.1, 9.0, 5.8, 7.8, 6.0, 8.8, 7.4, 5.5, 8.5, 6.8],
    "projects": [1, 3, 2, 4, 0, 3, 1, 4, 2, 0, 3, 1],
    "placed": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

X = df[["cgpa", "projects"]]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))