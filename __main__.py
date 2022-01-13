import os
import argparse # used for creating and passing Arguments for cmd
import numpy as np
import pandas as pd
from clf import *  # clf is a module containing class and fuction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA

# creating and passing Arguments for cmd
parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset',help='Dataset',type =str,default=None,required=True)
#parser.add_argument('-l','--label',type=str,default=None,required=True)
parser.add_argument('-p','--pca',help="pca_n_components",type=int,default=2,required=True)
args = parser.parse_args()


print("reading the dataset")
df = pd.read_csv(args.dataset)


print("spliting the dataset into idv and dv")
#x = df.drop(labels=args.label,axis=1)
#y = df[args.label]
x = df.iloc[:,:-1]
y = df.iloc[:,-1]


print("splitting the datset into train and test splits")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# Feature Scalling
print("Scalling the Dataset")
min = MinMaxScaler()
std = StandardScaler()
min.fit(x_train)
s_xtrain = min.transform(x_train)
s_xtest = min.transform(x_test)


#PCA(Principal Component Analysis)
print("Calculating Principal component analysis")
pca = PCA(n_components=args.pca)
p_xtrain = pca.fit_transform(s_xtrain)
p_xtest = pca.transform(s_xtest)


cl = Classifier(train_x=s_xtrain,train_y=y_train,test_x=s_xtest,test_y=y_test)# calling class "Classifiers" from clf.py module
pravalance_rate = cl.pravalance_rate()
train_results = cl.train() # calling train() function from "Classifier" Class in "clf" module
test_results = cl.test()
confusion_matrix_train = cl.confusion_matrix_train()
confusion_matrix_test = cl.confusion_matrix_test()
classification_report_train = cl.classification_report_train()
classification_report_test = cl.classification_report_test()
print("Pravalance Rate for training and testing")
print(pravalance_rate)
print("Training results...")
print(train_results)
print("\n")
print("Testing results...")
print(test_results)
print("\n")
print("Confusion Matrix for Training...")
print(confusion_matrix_train)
print("\n")
print("Confusion Matrix for Testing...")
print(confusion_matrix_test)
classification_report_train.to_csv("Classification_report_train")
classification_report_test.to_csv("Classification_report_test")
