import pandas as pd

df = pd.read_csv('dataset_1.csv', sep=',', encoding='utf-8')
X = df.drop(columns=['Result'])
y = df.Result

# Разделяем датасет на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.DataFrame(data=pd.concat([X_train, y_train], axis=1))
test_df = pd.DataFrame(data=pd.concat([X_test, y_test], axis=1))

train_df.to_csv('train_1.csv', sep=',', encoding='utf-8', index=None)
test_df.to_csv('test_1.csv', sep=',', encoding='utf-8', index=None)