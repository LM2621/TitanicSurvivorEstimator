import pandas as pd

#Read the data
train = pd.read_csv("/train.csv")
test = pd.read_csv("/test.csv")

#Create a dataframe from the data, leaving some columns (due to irrelevancy or many NaN values)
relevantFeatures = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
relevantFeaturesSubTarget = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
dfTrain = train[relevantFeatures]
dfTest = test[relevantFeaturesSubTarget]

#Impute mean values och nearest neighbour
pd.options.mode.chained_assignment = None

meanAge = dfTrain["Age"].mean()
dfTrain.Age = dfTrain.Age.fillna(meanAge)
meanAge = dfTest["Age"].mean()
dfTest.Age = dfTest.Age.fillna(meanAge)

dfTrain["Fare"] = dfTrain["Fare"].interpolate(method='nearest')
dfTest["Fare"] = dfTest["Fare"].interpolate(method='nearest')

#Save the target variable and remove the target variable column from the training data

targetTrain = dfTrain.Survived
dfTrain = dfTrain.drop("Survived", axis=1)

#One hot encode the categorical data

dfTrain = pd.get_dummies(dfTrain, columns=["Sex","Embarked"])
dfTest = pd.get_dummies(dfTest, columns=["Sex","Embarked"])

#1st model: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Crossvalidation of LogReg to give an idea of accuracy
logReg = LogisticRegression(max_iter=1000)
crossValidation = cross_val_score(logReg, dfTrain, targetTrain, cv=5)
print(crossValidation)
print(crossValidation.mean())

#Tune Log Reg hyperparameters by using gridsearch
from sklearn.model_selection import GridSearchCV

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet',],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000],
    'multi_class': ['ovr', 'multinomial'],
    'tol': [1e-4, 1e-3, 1e-2],
    'class_weight': [None, 'balanced'],
    'random_state': [None, 42, 2021]
}

gridsearch = GridSearchCV(logReg, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
gridsearchedLogReg = gridsearch.fit(dfTrain, targetTrain)

print("Logistic regression")
print('Best Score: ' + str(gridsearchedLogReg.best_score_))

#2nd model: Random forest classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)

#Crossvalidation of Random Forest to give an idea of accuracy
crossValidation = cross_val_score(rfc, dfTrain, targetTrain, cv=5)
print(crossValidation)
print(crossValidation.mean())

# Tune Random Forest hyperparameters by using gridsearch
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [5, 10, 15, 20, None],
    'max_features': ['sqrt', 'log2', 0.5],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

gridsearchedRfc = GridSearchCV(rfc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
gridsearchedRfc.fit(dfTrain, targetTrain)

print("Random Forest Classifier")
print('Best Score: ' + str(gridsearchedRfc.best_score_))

#3rd model: XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state =1)

#Crossvalidation of XGB to give an idea of accuracy
cv = cross_val_score(xgb,dfTrain,targetTrain,cv=5)
print(cv)
print(cv.mean())

#Tune XGB hyperparameters by using gridsearch

param_grid1 = {
    'n_estimators': [20, 50, 100, 250],
    'colsample_bytree': [0.2, 0.5, 0.7, 0.8],
    'max_depth': [2, 5, 10, 15],
    'subsample': [0.5,0.6,0.7, ],
    'learning_rate':[.01,0.1,0.2,0.3,0.5,],
    'gamma':[0,.01,.1,1,10],
    'sampling_method': ['uniform', 'gradient_based']}

gridsearchedXGBoost = GridSearchCV(xgb, param_grid = param_grid1, cv = 5, verbose = True, n_jobs = -1)
gridsearchedXGBoost.fit(dfTrain, targetTrain);

print("XGBoost")
print('Best Score: ' + str(gridsearchedXGBoost.best_score_))

#4th model: SVC
from sklearn.svm import SVC

svc = SVC()

#Crossvalidation of SVC to give an idea of accuracy
cv = cross_val_score(svc,dfTrain,targetTrain,cv=5)
print(cv)
print(cv.mean())

#Tune SVC hyperparameters by using gridsearch

param_grid = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5],
                                  'C': [.1, 1, 10, 100, 1000]}]

gridsearchedSVC = GridSearchCV(svc, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
gridsearchedSVC.fit(dfTrain, targetTrain)

gridsearchedSVC=SVC(C= 1, gamma= 0.1, kernel= 'rbf', probability=True)

print("SVC")
print('Best Score: ' + str(gridsearchedSVC.best_score_))

#Use a Voting Classifiers consisting of the tuned models as final ensemble models
from sklearn.ensemble import VotingClassifier

VotingClfSoft = VotingClassifier(estimators=[('LogReg', gridsearchedLogReg),('RFC',gridsearchedRfc),('XGBoost',gridsearchedXGBoost),('SVC',gridsearchedSVC)],voting='soft')
print("Soft voting classifier")
print('voting_clf_soft :',cross_val_score(VotingClfSoft,dfTrain,targetTrain,cv=5))
print('voting_clf_soft mean :',cross_val_score(VotingClfSoft,dfTrain,targetTrain,cv=5).mean())

VotingClfHard = VotingClassifier(estimators=[('LogReg', gridsearchedLogReg),('RFC',gridsearchedRfc),('XGBoost',gridsearchedXGBoost),('SVC',gridsearchedSVC)],voting='hard')
print("Hard voting classifier")
print('voting_clf_hard :',cross_val_score(VotingClfHard,dfTrain,targetTrain,cv=5))
print('voting_clf_hard mean :',cross_val_score(VotingClfHard,dfTrain,targetTrain,cv=5).mean())

VotingClfHard.fit(dfTrain,targetTrain)
results1 = VotingClfHard.predict(dfTest)
VotingClfSoft.fit(dfTrain,targetTrain)
results2 = VotingClfHard.predict(dfTest)

#Write results to files ready for submission
from datetime import date
import os

date = date.today();
date = date.strftime("%Y-%m-%d")  # Example format: YYYY-MM-DD
current_file_path = __file__
current_file_name = os.path.basename(current_file_path)  # Extract the name of the file from the path
filepath1 = current_file_path + current_file_name + date + "_1" + ".csv"
filepath2 = current_file_path + current_file_name + date + "_2" + ".csv"

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived': results1}).to_csv(filepath1, index=False)
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived': results2}).to_csv(filepath2, index=False)

print("Done")

