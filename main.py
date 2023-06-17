import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# read dataset
df = pd.read_excel('data.xlsx')


# display dataset
print(df.columns, "\n\n")
print(df.head(), "\n\n")
print(df.tail(), "\n\n")


# print data details
print("details = \n", df.describe(), '\n\n')


# drop unneeded columns
df = df.drop('Pclass', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)


# convert categorical data to numeric
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Cabin'] = le.fit_transform(df['Cabin'])
df['Embarked'] = le.fit_transform(df['Embarked'])


# fill missing values with mean
df = df.fillna(df.mean())


# split features and result
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(x, "\n\n")
print(y, "\n\n")


# normalize the data by decimal scaling
arr = np.array(x)
scale = np.max(np.abs(arr))
arr = arr / scale
print("data after normalization = \n", arr, '\n\n')


# split the data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
print('train data: \n', x_train, '\n\n')
print('train results: \n', y_train, '\n\n')

print('test data: \n', x_test, '\n\n')
print('test results: \n', y_test, '\n\n')


# implement naiive base
model = GaussianNB()
model.fit(x_train, y_train)


print('score = ', model.score(x_test, y_test), '\n\n')
print('predection = ', model.predict(x_test[0:10]), '\n\n')
print('cross validation score = ', cross_val_score(GaussianNB(), x_train, y_train, cv=2), '\n\n')