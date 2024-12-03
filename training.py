import pandas as pd

train_df = pd.read_csv('train_1.csv', sep=',', encoding='utf-8')
X = train_df.drop(columns='Result')
y = train_df.Result

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

# Токенизатор текстовых данных для обучения классификатора
cv = CountVectorizer(decode_error='ignore')
# Токенизатор для нескольких колонок
vectorizer = ColumnTransformer([
    ('vec_txt', CountVectorizer(decode_error='ignore'), 'Text'),
    ('vec_annot', CountVectorizer(decode_error='ignore'), 'Annotaion')
])
# Токенизация данных
X_train = vectorizer.fit_transform(X)
y_train = y

# Классификация данных
clf = SVC()
# pipeline = make_pipeline(vectorizer, clf)
text_clf = Pipeline([('vect' , vectorizer),
                     ('clf' , SVC())
                    ])

# Подбор оптимальных параметров модели
params = {
        'clf__C' : [1, 10, 100],
        #'kernel' : ['poly', 'linear', 'sigmoid', 'rbf', 'precomputed'],
        'clf__gamma' : ['scale', 'auto']
}
# Кросс-валидация
cross_val = KFold(shuffle=True, random_state=21)
# Инициализация классификатора с подбором параметров
random_cv_clf = GridSearchCV(estimator=text_clf, param_grid=params, cv=cross_val)
# Обучение
print('Fitting classifier...')
random_cv_clf.fit(X, y)
# Выбор классификатора с лучшими параметрами
best_clf = random_cv_clf.best_estimator_
with open('best_params_1.txt', 'w') as file:
    file.write(str(random_cv_clf.best_params_))
print('Classifier has been fitted.')

# Сохранение модели
from joblib import dump
dump(best_clf, 'best_clf_1.joblib')
