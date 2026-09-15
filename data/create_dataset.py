from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()

df = pd.DataFrame(digits.data)
df["target"] = digits.target

df.to_csv("data/digits.csv", index=False)

print("Dataset saved successfully!")
print("Shape:", df.shape)