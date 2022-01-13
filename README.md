# Auto_Ml_Application
Terminal application to build multiple machine learning classification models and get metrics for any given dataset.

This project consist of 
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K Nearest Neighbors Classifier
5. Gaussian Naive Bayes Classifier
6. AdaBoost Classifier
7. GradientBoostingClassifier

Working of Project :

Step 1 : Data Set is read from the folder using ArgumentParser in terminal
Step 2 : The dats set is divided into independent variables and dependent variables are
Step 3 : Further the data set is divided into train test splits
Step 4 : Feature Scalling is done using MinMaxScaler
Step 5 : PCA is done and principle components can be used to train the model(optional)
Step 6 : pravalance rate is printed for Dependent train and test data
Step 7 : All the above metioned models are trained and accuracy score is printed
Step 8 : Same is done for testing data
Step 9 : Confusion Matrix is printed for traning and testing data
Step 10 : Classification report for train and test data is exported to a CSV File in the same folder


To run the application

Step 1 : open Command Prompt in the project folder 
Step 2 : Activate Python environment
Step 3 : To run app type : python __main__.py -d "dataset_name.csv"

Project Objective:
    To reduce the time taken to build each and every model and check their metrics