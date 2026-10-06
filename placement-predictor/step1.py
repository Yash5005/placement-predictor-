import pandas as pd 
data = {
    "cgpa":[6.5, 8.2, 7.1, 9.0, 5.8, 7.8,],
    "projects":[1, 3, 2, 4, 0, 3],
    "placed":[0, 1, 0, 1, 0, 1],
}

df = pd.DataFrame(data)
print(df)
print(df.shape)