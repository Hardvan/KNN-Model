import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Getting Data
df = pd.read_csv("KNN_Project_Data")
print(df.head())
print()
print(df.describe())
print()
print(df.info())
print()

# EDA
sns.pairplot(df, hue = "TARGET CLASS", palette="magma")

# Standardizing Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop(["TARGET CLASS"], axis = 1))
scaled_features = scaler.transform(df.drop(["TARGET CLASS"], axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
print(df_feat.head())

# Training Model
from sklearn.model_selection import train_test_split
X = df_feat
y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predictions and Evaluation
predictions = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print("WITH K = 1\n")
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))

# Choosing Best K Value
error_rates = []
for i in range(1,41):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train, y_train)
    pred_i = knn_i.predict(X_test)
    error_rates.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,41), error_rates, ls = "dashed", marker = "o", markerfacecolor = "red", ms = 10)
plt.title("Error Rate v/s K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")

# K = 20 is best
knn_best = KNeighborsClassifier(n_neighbors=20)
knn_best.fit(X_train, y_train)
pred_best = knn_best.predict(X_test)
print("WITH K = 20\n")
print(confusion_matrix(y_test, pred_best))
print()
print(classification_report(y_test, pred_best))
print()


























