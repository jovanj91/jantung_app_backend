import itertools
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import pickle
import joblib

df = pd.read_csv('M1F2_PSAX.csv')
X = df.drop('CLASS', axis=1)
y = df['CLASS']
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

model = svm.SVC(kernel='linear', gamma='auto')

print("\t####---Split Data (80%)---####\n")
loo = LeaveOneOut()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
report = classification_report(y_test, yhat, labels=[0, 1], output_dict=True)
print(report)
cohen_score = cohen_kappa_score(y_test, yhat)
print(f"Train set Accuracy: {metrics.accuracy_score(y_train, model.predict(X_train)) * 100:.2f}%" )
print(f"Test set Accuracy: {metrics.accuracy_score(y_test, yhat) * 100:.2f}%" )
print(f"Kappa Score:{cohen_score * 100:.2f} \n")
matrix = confusion_matrix(y_test, yhat)
print(matrix,"\n")
scores = cross_val_score(model,X_train,y_train,cv=3,scoring='accuracy')
print (scores)
mean_accuracy = scores.mean()
std_accuracy = scores.std()
print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {std_accuracy * 1:.3f}")

df_report = pd.DataFrame(report).transpose()

fpr, tpr, thresholds = roc_curve(y_test, yhat)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, yhat)
average_precision = average_precision_score(y_test, yhat)

train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

flow_choice = input("Enter 1 to Generate Model or else To Decline: ")
flow_choice = int(flow_choice)

if flow_choice == 1:
    model = svm.SVC(kernel='linear', gamma='auto')
    model.fit(X, y)
    filename = 'modelSVM'
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X, y)
    print(result)
else:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap="YlGnBu")
    plt.title('Classification Report')
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc="best")
    plt.show()