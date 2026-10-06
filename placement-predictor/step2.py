import pandas as pd
from sklearn.linear_model import LogisticRegression
data = {
    "cgpa": [6.5, 8.2, 7.1, 9.0, 5.8, 7.8],
    "projects": [1, 3, 2, 4, 0, 3],
    "placed": [0, 1, 0, 1, 0, 1],
}
df = pd.DataFrame(data)
X = df[["cgpa", "projects"]]
y = df["placed"]  

model = LogisticRegression()
model.fit(X, y)

new_student = pd.DataFrame([{"cgpa": 8.0, "projects": 2}])
print("Prediction:", model.predict(new_student))
print("Chance of placement:", model.predict_proba(new_student)[0][1])