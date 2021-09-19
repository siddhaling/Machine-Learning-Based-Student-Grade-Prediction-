import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
data = pd.read_csv('student-mat.csv')
data.shape
data.isnull().values.any()
data['GAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3
results = []
names = []
error= []
precision=[]
True_Rate=[]
names.append("Decision trees")
names.append("SVM")
names.append("NN")
names.append("NB")
names.append("RF")
names.append("LR")
def define_grade(df):
    # Create a list to store the data
    grades = []

    # For each row in the column,
    for row in df['GAvg']:
        # if more than a value,
        if row >= (0.9 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('A')
        # else, if more than a value,
        elif row >= (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('B')
        # else, if more than a value,
        elif row < (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('C')   
    # Create a column from the list
    df['grades'] = grades
    return df

data = define_grade(data)


 
data.drop(["school","age"], axis=1, inplace=True)

# for yes / no values:
d = {'yes': 1, 'no': 0}
data['schoolsup'] = data['schoolsup'].map(d)
data['famsup'] = data['famsup'].map(d)
data['paid'] = data['paid'].map(d)
data['activities'] = data['activities'].map(d)
data['nursery'] = data['nursery'].map(d)
data['higher'] = data['higher'].map(d)
data['internet'] = data['internet'].map(d)
data['romantic'] = data['romantic'].map(d)

 d = {'F': 1, 'M': 0}
data['sex'] = data['sex'].map(d)

# map the address data
d = {'U': 1, 'R': 0}
data['address'] = data['address'].map(d)

# map the famili size data
d = {'LE3': 1, 'GT3': 0}
data['famsize'] = data['famsize'].map(d)

# map the parent's status
d = {'T': 1, 'A': 0}
data['Pstatus'] = data['Pstatus'].map(d)

# map the parent's job
d = {'teacher': 0, 'health': 1, 'services': 2,'at_home': 3,'other': 4}
data['Mjob'] = data['Mjob'].map(d)
data['Fjob'] = data['Fjob'].map(d)

# map the reason data
d = {'home': 0, 'reputation': 1, 'course': 2,'other': 3}
data['reason'] = data['reason'].map(d)

# map the guardian data
d = {'mother': 0, 'father': 1, 'other': 2}
data['guardian'] = data['guardian'].map(d)

# map the grades data
d = {'C': 0, 'B': 1, 'A': 2}
data['grades'] = data['grades'].map(d)

student_features = data.columns.tolist()
student_features.remove('grades') 
student_features.remove('GAvg') 
student_features.remove('G1') 
student_features.remove('G2') 
student_features.remove('G3') 
student_features

X = data[student_features].copy()

y=data[['grades']].copy()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=200)

#Decision tree classification

grade_classifier = tree.DecisionTreeClassifier(max_leaf_nodes=len(X.columns), random_state=0)
#grade_classifier.fit(X_train, y_train)

#from pydotplus import graphviz
#dot_data = StringIO()  
#tree.export_graphviz(grade_classifier, out_file=dot_data,feature_names=student_features)
#graph = graphviz.graph_from_dot_data(dot_data)
#graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())

#predictions = grade_classifier.predict(X_test)
#from sklearn.metrics import roc_curve
#from matplotlib import pyplot
#fpr,tpr,thresholds=roc_curve(y_test,predictions)
#pyplot.plot(fpr,tpr,marker='.')
#pyplot.show()

print("Classification using Decision Trees:")
print('\n')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')
print('\n')
# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=grade_classifier,learning_rate=1)

# Train Adaboost Classifer
abc.fit(X_train, y_train)

#Predict the response for test dataset
predictions = abc.predict(X_test)
print("Accuracy using Decision trees:",accuracy_score(y_true = y_test, y_pred = predictions,normalize='true'))
Accuracy_Score = accuracy_score(y_true = y_test, y_pred = predictions,normalize='true')
results.append(Accuracy_Score)
Precision_Score = precision_score(y_test, predictions,  average="macro")
precision.append(Precision_Score)
Recall_Score = recall_score(y_test, predictions,  average="macro")
F1_Score = f1_score(y_test, predictions,  average="macro")
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
error_score=metrics.mean_absolute_error(y_test, predictions)
error.append(error_score)
c_matrix=confusion_matrix(y_test, predictions)
print(c_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(c_matrix, annot=True)
plt.show()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=predictions))
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = c_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("True positive rate:",TPR[0])
print("True Negative Rate:",TNR[0])
print("False Positive Rate:",FPR[0])
print("False Negative Rate:",FNR[0])
True_Rate.append(TPR[0])
print('\n')

print("Classification using SVM:")

from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
clf = []
clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)     

predicted_label = clf.predict(X_test)

Accuracy_Score = accuracy_score(y_true = y_test, y_pred = predicted_label,normalize='true')
print("Accuracy using SVM is:",Accuracy_Score)
results.append(Accuracy_Score)
Precision_Score = precision_score(y_test, predicted_label,  average="macro")
precision.append(Precision_Score)
Recall_Score = recall_score(y_test, predicted_label,  average="macro")
F1_Score = f1_score(y_test, predicted_label,  average="macro")
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted_label))
error_score=metrics.mean_absolute_error(y_test, predicted_label)
error.append(error_score)
c_matrix=confusion_matrix(y_test, predicted_label)
print(c_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(c_matrix, annot=True)
plt.show()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=predicted_label))
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = c_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("True positive rate:",TPR[0])
print("True Negative Rate:",TNR[0])
print("False Positive Rate:",FPR[0])
print("False Negative Rate:",FNR[0])
True_Rate.append(TPR[0])
print('\n')

#Nearest Neighbours


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(7)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
Accuracy_Score = accuracy_score(y_true = y_test, y_pred = y_predict,normalize='true')
results.append(Accuracy_Score)
print("Accuracy using K Neighbours is:",Accuracy_Score)
Precision_Score = precision_score(y_test, y_predict,  average="macro")
Recall_Score = recall_score(y_test, y_predict,  average="macro")
F1_Score = f1_score(y_test, y_predict,  average="macro")
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
error_score=metrics.mean_absolute_error(y_test, y_predict)
error.append(error_score)
c_matrix=confusion_matrix(y_test, y_predict)
print(c_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(c_matrix, annot=True)
plt.show()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_predict))
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = c_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("True positive rate:",TPR[0])
print("True Negative Rate:",TNR[0])
print("False Positive Rate:",FPR[0])
print("False Negative Rate:",FNR[0])
True_Rate.append(TPR[0])
precision.append(Precision_Score)
print('\n')

#Naive Bayesian

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,y_train)
y_score = nb.predict(X_test)
Accuracy_Score = accuracy_score(y_true = y_test, y_pred = y_score,normalize='true')
results.append(Accuracy_Score)
print("Accuracy using Naive Bayesian is:",Accuracy_Score)
Precision_Score = precision_score(y_test, y_score,  average="macro")
Recall_Score = recall_score(y_test, y_score,  average="macro")
F1_Score = f1_score(y_test, y_score,  average="macro")
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_score))
error_score=metrics.mean_absolute_error(y_test, y_score)
error.append(error_score)
c_matrix=confusion_matrix(y_test, y_score)
print(c_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(c_matrix, annot=True)
plt.show()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_score))
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = c_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("True positive rate:",TPR[0])
print("True Negative Rate:",TNR[0])
print("False Positive Rate:",FPR[0])
print("False Negative Rate:",FNR[0])
True_Rate.append(TPR[0])
precision.append(Precision_Score)


print('\n')
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
Accuracy_Score = accuracy_score(y_true = y_test, y_pred = y_pred,normalize='true')
results.append(Accuracy_Score)
print("Accuracy using Random Forest is:",Accuracy_Score)
Precision_Score = precision_score(y_test, y_pred,  average="macro")
Recall_Score = recall_score(y_test, y_pred,  average="macro")
F1_Score = f1_score(y_test, y_pred,  average="macro")
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
error_score=metrics.mean_absolute_error(y_test, y_pred)
error.append(error_score)
c_matrix=confusion_matrix(y_test, y_pred)
print(c_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(c_matrix, annot=True)
plt.show()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_pred))
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = c_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("True positive rate:",TPR[0])
print("True Negative Rate:",TNR[0])
print("False Positive Rate:",FPR[0])
print("False Negative Rate:",FNR[0])
True_Rate.append(TPR[0])
precision.append(Precision_Score)
print('\n')
#Logistic Regression


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)
y_prediction = LR.predict(X_test)
Accuracy_Score = accuracy_score(y_true = y_test, y_pred = y_prediction,normalize='true')
results.append(Accuracy_Score)
print("Accuracy using Logistic Regression is:",Accuracy_Score)
Precision_Score = precision_score(y_test, y_prediction,  average="macro")
Recall_Score = recall_score(y_test, y_prediction,  average="macro")
F1_Score = f1_score(y_test, y_prediction,  average="macro")
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_prediction))
error_score=metrics.mean_absolute_error(y_test, y_prediction)
error.append(error_score)
cnf_matrix=confusion_matrix(y_test, y_prediction)
print(cnf_matrix)
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(cnf_matrix, annot=True)
plt.show()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_prediction))

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("True positive rate:",TPR[0])
print("True Negative Rate:",TNR[0])
print("False Positive Rate:",FPR[0])
print("False Negative Rate:",FNR[0])

True_Rate.append(TPR[0])
precision.append(Precision_Score)

#Comparing the accuracies using bar graph
y_pos = np.arange(len(names))

plt.bar(y_pos, results, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('Accuracy')
plt.title('Classification technique')

plt.show()

#Comparing the precision using bar graph
y_pos = np.arange(len(names))

plt.bar(y_pos, precision, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('Precision')
plt.title('Classification technique')

plt.show()

#Comparing error using bar graph
y_pos = np.arange(len(names))

plt.bar(y_pos, error, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('Mean Absolute Error')
plt.title('Classification technique')

plt.show()

#Comparing TPR using bar graph
y_pos = np.arange(len(names))

plt.bar(y_pos, True_Rate, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('True Positive Rate')
plt.title('Classification technique')

plt.show()