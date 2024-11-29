import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

url = "https://docs.google.com/spreadsheets/d/1XloteWzWK7lsgDeJ_WZr2K8SkCYQS3pONPFyddp6lf4/export?format=csv"

df = pd.read_csv(url)

sex_mapping = {'Nam': 0, 'Nữ': 1}
df['2. Giới tính của bạn là ?'] = df['2. Giới tính của bạn là ?'].map(sex_mapping)
cp_mapping = {'Không': 0, 'Có': 1}
df['3. Bạn có triệu trứng đau ngực không ?'] = df['3. Bạn có triệu trứng đau ngực không ?'].map(cp_mapping)
fbs_mapping = {'Bình thường': 0, 'Cao': 1}
df['6. Đường huyết của bạn là bao nhiêu ?'] = df['6. Đường huyết của bạn là bao nhiêu ?'].map(fbs_mapping)
restecg_mapping = {'Bình thường': 0, 'Không bình thường': 2, 'Có dấu hiệu bất thường':1}
df['7. Kết quả điện tâm đồ tĩnh của bạn là gì ?'] = df['7. Kết quả điện tâm đồ tĩnh của bạn là gì ?'].map(restecg_mapping)
exang_mapping = {'Không': 0, 'Có': 1}
df['9. Bạn có bị đau tim khi tập thể dục không ?'] = df['9. Bạn có bị đau tim khi tập thể dục không ?'].map(exang_mapping)
thal_mapping = {'Không có': 0, 'Bình thường': 1, 'Bất thường':2, 'Bất thường nghiêm trong':3}
df['10. Bạn có bị bệnh thiếu máu tán bẩm sinh (Thalassemia) không ? Nếu có thì thuộc loại gì?'] = df['10. Bạn có bị bệnh thiếu máu tán bẩm sinh (Thalassemia) không ? Nếu có thì thuộc loại gì?'].map(thal_mapping)

new_column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'thal']



df = df.drop('Timestamp', axis=1)

df.columns = new_column_names

df.to_csv('/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/data2.csv', index=False)


train_data = np.genfromtxt('/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/heart.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13))

x_train = train_data[:, :-1]
y_train = train_data[:, -1]

# test_data = np.genfromtxt('/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/data2.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
# Loại bỏ hàng có giá trị NaN trong x_train và y_train
mask = ~np.isnan(x_train).any(axis=1)
x_train = x_train[mask]
y_train = y_train[mask]
# test_data = test_data[mask]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# y_pred = knn.predict(test_data)

# test_data_labeled = np.column_stack((test_data, y_pred))

# print("Kết quả dự đoán trên dữ liệu test: \n", test_data_labeled)

# accuracy = accuracy_score(y_train, y_pred)
# print("Độ chính xác của mô hình KNN là:", accuracy)

# df = pd.DataFrame(test_data_labeled, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'thal', 'target'])
# df['age'] = df['age'].astype(int)  # Chuyển đổi cột 'age' sang kiểu dữ liệu số nguyên
# df['target'] = df['target'].astype(int)  # Chuyển đổi cột 'target' sang kiểu dữ liệu số nguyên
# pd.crosstab(df.age, df.target).plot(kind="bar", figsize=(16, 6))
# plt.title('Heart Disease Frequency for Ages')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()