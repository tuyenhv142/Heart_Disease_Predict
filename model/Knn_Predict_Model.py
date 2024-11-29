import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


train_data = np.genfromtxt('/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/heart.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13))
x_train = train_data[:, :-1]
y_train = train_data[:, -1]

# Loại bỏ hàng có giá trị NaN trong x_train và y_train
mask = ~np.isnan(x_train).any(axis=1)
x_train = x_train[mask]
y_train = y_train[mask]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Lưu mô hình sau khi huấn luyện
joblib.dump(knn, "/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/knn_model.sav")
print("Mô hình đã được lưu tại: knn_model.sav")

# Tải lại mô hình đã lưu
knn_loaded = joblib.load("/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/knn_model.sav")
print("Mô hình đã được tải lại.")

