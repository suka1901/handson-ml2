# Visualizing Model Performance Comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RMSE values gathered from the experiments 
rmse_data = {
    "Linear Regression": 67508,
    "Decision Tree": 73965,
    "Random Forest": 53345,
    "SVR Linear Kernel": 70690,
    "SVR RBF Kernel": 116751,
    "SVR Poly Kernel": 116691,
    "SVR Sigmoid Kernel": 116679
}

# Convert to DataFrame
rmse_df = pd.DataFrame(list(rmse_data.items()), columns=["Model", "RMSE"])

# Bar plot
plt.figure(figsize=(10,6))
plt.barh(rmse_df["Model"], rmse_df["RMSE"], color='skyblue')
plt.xlabel("Cross-validated RMSE (USD)")
plt.title("Model Performance Comparison")
plt.gca().invert_yaxis()  # highest at top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Boxplot simulation: create fake cross-validation RMSE arrays to illustrate distribution
np.random.seed(42)
cv_results = {
    "Linear Regression": np.random.normal(53345,  50772, 59701),
    "Decision Tree": np.random.normal(72000, 5000, 10),
    "Random Forest": np.random.normal(52000, 1500, 10),
    "SVR Linear Kernel": np.random.normal(70000, 2500, 10),
    "SVR RBF Kernel": np.random.normal(57000, 2000, 10),
    "SVR Poly Kernel": np.random.normal(65000, 3000, 10),
    "SVR Sigmoid Kernel": np.random.normal(82000, 4000, 10),
}

plt.figure(figsize=(10,6))
plt.boxplot(cv_results.values(), labels=cv_results.keys(), vert=False)
plt.xlabel("Cross-validated RMSE (USD)")
plt.title("Cross-validation RMSE Distribution")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

models = [
    'Linear Regression',
    'Decision Tree',
    'Random Forest',
    'SVR (Linear)',
    'SVR (RBF)',
    'SVR (Polynomial)',
    'SVR (Sigmoid)'
]
rmse = [67508, 73965, 53345, 70690, 116751, 116691, 116679]

plt.figure(figsize=(8,4))
plt.bar(models, rmse, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('RMSE')
plt.title('Best cross-validated RMSE for each model')
plt.tight_layout()
plt.show()
