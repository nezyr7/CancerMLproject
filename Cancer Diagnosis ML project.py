import pandas as pd
from sklearn import preprocessing
data_set=pd.read_csv(r'C:\Users\Tech\Downloads\cancer.csv')
print(data_set.shape)
print(data_set.head(0))
print(data_set.nunique())
print(data_set.dtypes)
print(data_set.isnull().sum())
d_types=data_set.dtypes
for i in range(data_set.shape[1]):
    if d_types[i] == 'object':
        Pr_data = preprocessing.LabelEncoder()
        data_set[data_set.columns[i]] = \
        Pr_data.fit_transform(data_set[data_set.columns[i]])
        print("Column index = ", i)
        print(Prdata.classes)
scaler = preprocessing.MinMaxScaler()
Scaled_data = scaler.fit_transform(data_set)
Scaled_data = \
pd.DataFrame(Scaled_data, columns=data_set.columns)
r=Scaled_data.corr()
print(r)
import seaborn as sns
import matplotlib.pyplot as plt
r=Scaled_data.corr()
sns.heatmap(r, annot=True)
plt.show()
sns.pairplot(Scaled_data)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset = pd.read_csv(r'C:\Users\Tech\Downloads\cancer.csv')
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000)
model.evaluate(x_test, y_test)