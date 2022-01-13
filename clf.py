import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

class Classifier:
    def __init__(self,train_x=None,train_y=None,test_x=None,test_y=None):
        self.xtrain = train_x
        self.ytrain = train_y
        self.xtest = test_x
        self.ytest = test_y
        self.models=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GaussianNB(),KNeighborsClassifier(),AdaBoostClassifier(),GradientBoostingClassifier()]
        self.names=["Logistic Rrgression","Decision Tree","Random Forest","GausianNaive Bayes","K Nearest","AdaBoost","GradientBoost"]
    
    def pravalance_rate(self):
        print("Prevalance rate train")
        Prevalance_rate_train=self.ytrain.value_counts(normalize=True).mul(100).round(1)
        print("\n")
        print("Prevalance rate test")
        Prevalance_rate_test=self.ytest.value_counts(normalize=True).mul(100).round(1)
        pravalance_rate = Prevalance_rate_train+Prevalance_rate_test
        return Prevalance_rate_train,Prevalance_rate_test

    def train(self):
        train_results = pd.DataFrame(columns=['Models','train_Accuracy'])
        for name,model in zip(self.names,self.models):
            print((f'Training {name} model'))
            model.fit(self.xtrain,self.ytrain)
            train_acc=model.score(self.xtrain,self.ytrain)
            train_results = train_results.append({'Models':name,'train_Accuracy':train_acc},ignore_index=True)
        return train_results

    def test(self):
        test_results = pd.DataFrame(columns=['Models','test_Accuracy'])
        for name,model in zip(self.names,self.models):
            print((f'Testing {name} model'))
            model.fit(self.xtest,self.ytest)
            test_acc=model.score(self.xtest,self.ytest)
            test_results = test_results.append({'Models':name,'test_Accuracy':test_acc},ignore_index=True)
        return test_results

    def confusion_matrix_train(self):
        cons = pd.DataFrame(columns=['Models','Confusion_matrix'])
        for name,model in zip(self.names,self.models):
            model.fit(self.xtrain,self.ytrain)
            y_pred_train = model.predict(self.xtrain)
            con = confusion_matrix(y_pred_train,self.ytrain)
            cons = cons.append({'Models':name,'Confusion_matrix':con},ignore_index=True)
        return cons

    def confusion_matrix_test(self):
        cons_test = pd.DataFrame(columns=['Models','Confusion_matrix'])
        for name,model in zip(self.names,self.models):
            model.fit(self.xtest,self.ytest)
            y_pred_test = model.predict(self.xtest)
            con = confusion_matrix(y_pred_test,self.ytest)
            cons_test = cons_test.append({'Models':name,'Confusion_matrix':con},ignore_index=True)
        return cons_test

    def classification_report_train(self):
        cls_report = pd.DataFrame(columns=['Models','Classification_report_train'])
        for name,model in zip(self.names,self.models):
            model.fit(self.xtrain,self.ytrain)
            y_pred_train = model.predict(self.xtrain)
            report = classification_report(self.ytrain, y_pred_train)
            cls_report = cls_report.append({'Models':name,'Classification_report_train':report},ignore_index=True)
        return cls_report               

    def classification_report_test(self):
        cls_report = pd.DataFrame(columns=['Models','Classification_report_test'])
        for name,model in zip(self.names,self.models):
            model.fit(self.xtest,self.ytest)
            y_pred_test = model.predict(self.xtest)
            report = classification_report(self.ytest, y_pred_test)
            cls_report = cls_report.append({'Models':name,'Classification_report_test':report},ignore_index=True)
        return cls_report

    