import pandas as pd

test_df = pd.read_csv('test.csv', encoding='utf-8', sep=',')
X_test = test_df.drop(columns=['Result'])
y_test = test_df.Result

# Загрузка модели
from joblib import load
clf = load('best_clf.joblib')

# Валидация модели
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

y_pred = clf.predict(X_test)
acc = round(max(cross_val_score(clf, X_test, y_test, cv=5)), 3)
prec = round(precision_score(y_test, y_pred, pos_label='Yes'), 3)
rec = round(recall_score(y_test, y_pred, pos_label='Yes'), 3)
f1 = round(f1_score(y_test, y_pred, pos_label='Yes'), 3)
cm = confusion_matrix(y_test, y_pred)


y_pred_digits = [0] * len(y_pred)
y_test_digits = [0] * len(y_test)
for i in range(len(y_test)):
    if y_test[i] == 'Yes':
        y_test_digits[i] = 1
    if y_pred[i] == 'Yes':
        y_pred_digits[i] = 1
auc = round(roc_auc_score(y_test_digits, y_pred_digits), 3)
with open('validation.txt', 'w') as file:
    file.write(f'True positive: {cm[0][0]}\n'
               f'False positive: {cm[1][0]}\n'
               f'False negative: {cm[0][1]}\n'
               f'True negative: {cm[1][1]}\n'
               f'Accuracy: {acc}\n'
               f'Precision: {prec}\n'
               f'Recall: {rec}\n'
               f'F1: {f1}\n'
               f'AUC: {auc}')

import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test_digits, y_pred_digits)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()