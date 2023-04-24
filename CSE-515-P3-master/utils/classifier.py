from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def SVM(X, y):
    clf = LinearSVC() # 0.53
    # clf = SVC(kernel='linear', C=1, decision_function_shape='ovr') # 0.47
    # clf = SVC(kernel='linear', C=1, decision_function_shape='ovo') # 0.47
    # clf = SVC(kernel='rbf', C=1, decision_function_shape='ovr') # 0.37
    # clf = SVC(kernel='rbf', C=1, decision_function_shape='ovo') # 0.37
    # clf = SVC(kernel='poly', degree=1, decision_function_shape='ovo') # 0.28-0.29
    # clf = SVC(kernel='poly', degree=3, decision_function_shape='ovr') # 0.28
    # clf = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo') # 0.28
    clf.fit(X, y)
    return clf

def svm_predict(clf, data):
    return clf.predict(data)

def DT(X, y):
    clf = DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X, y)
    return clf

def dt_predict(clf, data):
    return clf.predict(data)

def get_confusion_matrix(ytest, ypred, classes):
    cm = confusion_matrix(ytest, ypred, labels=classes)
    FP = np.sum(cm, axis=0) - np.diag(cm)  
    FN = np.sum(cm, axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = np.sum(cm) - (FP + FN + TP)
    return cm, FP, FN, TP, TN

def get_miss_rate(FN, TP):
    # False negative rate
    FNR = FN/(TP+FN)
    return FNR

def get_fp_rate(FP, TN):
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    return FPR

def display_confusion_matrix(cm, classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp = disp.plot()
    plt.show()

def get_class_count(classes):
    class_count_map = {}
    for cls in classes:
        if cls in class_count_map:
            class_count_map[cls] = class_count_map[cls] + 1
        else:
            class_count_map[cls] = 1
    return class_count_map

def get_accuracy_score(y_test,predicted_y_test):
    return accuracy_score(y_test,predicted_y_test)