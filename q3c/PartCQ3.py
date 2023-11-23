import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_names = [
    "Sample code number", "Clump Thickness", "Uniformity of Cell Size", 
    "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
    "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"
]
df = pd.read_csv('data1.csv', names=column_names)
# df.to_csv("data1.csv", index=False)
# Handle missing values by replacing '?' with NaN and converting to numeric
# df.replace('?', pd.NA, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df=df.dropna()
# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]
print(y)
# Normalize the input features between 0 and 1
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# One-hot encode the target classes
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

print(X_normalized)  # This will print the shape of X_normalized
print(y_encoded.shape)     # This will print the shape of y_encoded

# Create a DataFrame with the normalized data
columns = list(X.columns) + ["Class_" + str(i) for i in range(y_encoded.shape[1])]
x_columns = list(X.columns)  # Column names for X_normalized
y_columns = ["Class_" + str(i) for i in range(y_encoded.shape[1])]  # Column names for y_encoded

print(columns)
df_normalized = pd.DataFrame(data=pd.concat([pd.DataFrame(X_normalized, columns=x_columns), pd.DataFrame(y_encoded, columns=y_columns)], axis=1))

# Save the normalized data to data.csv
df_normalized.to_csv("data.csv", index=False)

# Load the processed data
df_normalized = pd.read_csv("data.csv")
print(df_normalized)
# Separate features and target
X = df_normalized.drop(["Class_0", "Class_1"], axis=1)  # Exclude one of the one-hot encoded columns
y = df_normalized[["Class_0", "Class_1"]]  # Considering binary classification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define neural network models with different optimizers
mlp_adam = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42, solver='adam')
mlp_sgd = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42, solver='sgd')

# Fit models and make predictions
mlp_adam.fit(X_train, y_train)
mlp_sgd.fit(X_train, y_train)

y_train_pred_adam = mlp_adam.predict(X_train)
y_test_pred_adam = mlp_adam.predict(X_test)

y_train_pred_sgd = mlp_sgd.predict(X_train)
y_test_pred_sgd = mlp_sgd.predict(X_test)

# Evaluate performance
accuracy_train_adam = accuracy_score(y_train, y_train_pred_adam)
accuracy_test_adam = accuracy_score(y_test, y_test_pred_adam)
rmse_test_adam = mean_squared_error(y_test, y_test_pred_adam, squared=False)

accuracy_train_sgd = accuracy_score(y_train, y_train_pred_sgd)
accuracy_test_sgd = accuracy_score(y_test, y_test_pred_sgd)
rmse_test_sgd = mean_squared_error(y_test, y_test_pred_sgd, squared=False)

# Print performance metrics
print("Adam Optimizer:")
print(f"Accuracy (Train): {accuracy_train_adam:.4f}")
print(f"Accuracy (Test): {accuracy_test_adam:.4f}")
print(f"RMSE (Test): {rmse_test_adam:.4f}")

print("\nSGD Optimizer:")
print(f"Accuracy (Train): {accuracy_train_sgd:.4f}")
print(f"Accuracy (Test): {accuracy_test_sgd:.4f}")
print(f"RMSE (Test): {rmse_test_sgd:.4f}")

# Plot ROC curve and calculate AUC for Adam Optimizer
fpr_adam, tpr_adam, _ = roc_curve(y_test["Class_1"], mlp_adam.predict_proba(X_test)[:, 1])
roc_auc_adam = auc(fpr_adam, tpr_adam)

plt.figure(figsize=(8, 6))
plt.plot(fpr_adam, tpr_adam, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_adam:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Adam Optimizer')
plt.legend(loc="lower right")
plt.show()