import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    "Size": [500, 700, 900, 1100, 1300],
    "Price": [100000, 150000, 200000, 250000, 300000]
}

df = pd.DataFrame(data)

X = df[["Size"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# Prediction
predicted_price = model.predict([[1000]])
print("Predicted Price:", predicted_price[0])
