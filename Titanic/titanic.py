# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def analize_data():
    # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    print train_df.columns.values

    print train_df.info()

    # 则会判断哪些”列”存在缺失值
    #print train_df.isnull().any()

    print train_df.describe()
    print train_df.describe(include=['O'])

    print 'Pclass'
    df_temp1 = train_df[['Pclass', 'Survived']].groupby(by='Pclass', as_index=False).mean().sort_values(
        by='Survived', ascending=False)
    print df_temp1

    print 'Sex:'
    df_temp2 = train_df[['Sex', 'Survived']].groupby(by='Sex', as_index=False).mean().sort_values(
        by='Survived', ascending=False)
    print df_temp2

    print 'SibSp:'
    df_temp3 = train_df[['SibSp', 'Survived']].groupby(by='SibSp', as_index=False).mean().sort_values(
        by='Survived', ascending=False)
    print df_temp3

    print 'Parch:'
    df_temp4 = train_df[['Parch', 'Survived']].groupby(by='Parch', as_index=False).mean().sort_values(
        by='Survived', ascending=False)
    print df_temp4


def analize_by_pic():
    train_df = pd.read_csv('./train.csv')
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    plt.show()

    g2 = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    g2.map(plt.hist, 'Age', bins=20)
    #g2.add_legend()
    plt.show()

    g3 = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
    g3.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    g3.add_legend()
    plt.show()

    g4 = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    g4.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    g4.add_legend()
    plt.show()


def feature_manipulate():
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    #  print pd.crosstab(train_df['Title'], train_df['Sex'])

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
                                                     'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    #  print combine[0][['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)
        dataset['Title'].fillna(0)


    #  print combine[0][['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    train_df = combine[0].drop(['Name', 'PassengerId'], axis=1)
    test_df = combine[1].drop(['Name'], axis=1)
    combine = [train_df, test_df]

    #  print train_df.shape, test_df.shape

    sex_mapping = {'female': 1, 'male': 0}

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)

    # complete the age's nan with median

    age_df = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):

                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
                age_guess = guess_df.median()
                age_df[i, j] = int(age_guess/0.5 + 0.5)*0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset['Sex'] == i) &
                            (dataset['Pclass'] == j+1), 'Age'] = age_df[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    #combine[0]['AgeBand'] = pd.cut(combine[0]['Age'], 5)
    #print combine[0][['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    for dataset in combine:
        dataset['Age*Class'] = dataset['Age']*dataset['Pclass']

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    #print combine[0][['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values('Survived', ascending=False)

    for dataset in combine:
        dataset['isAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1
    #  print combine[0][['isAlone', 'Survived']].groupby('isAlone', as_index=False).mean().sort_values('Survived', ascending=False)

    train_df = combine[0].drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
    test_df = combine[1].drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

    freq_port = train_df.Embarked.dropna().mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)

    combine = [train_df, test_df]
    embarked_mapping = {'C': 0, 'S': 1, 'Q': 2}

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

    #  combine[0]['FareBand'] = pd.qcut(combine[0]['Fare'], 4)
    #  print combine[0][['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean().sort_values('FareBand', ascending=True)

    combine[1]['Fare'].fillna(combine[1]['Fare'].dropna().median(), inplace=True)
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31.0, 'Fare'] = 3

        dataset['Fare'] = dataset['Fare'].astype(int)
    return combine[0], combine[1]


def model():
    train_df, test_df = feature_manipulate()
    print train_df.head(10)
    print test_df.head(10)

    x_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    x_test = test_df.drop('PassengerId', axis=1).copy()
    print x_train.shape, y_train.shape, x_test.shape

    '''
    逻辑回归
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    acc_log = round(logreg.score(x_train, y_train)*100, 2)
    print acc_log

    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    print coeff_df
    coeff_df.columns = ['Features']
    print logreg.coef_
    coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

    print coeff_df.sort_values('Correlation', ascending=False)
    
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print 'svc', round(svc.score(x_train, y_train)*100, 2)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print 'knn:', round(knn.score(x_train, y_train)*100, 2)

    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    y_pred = gaussian.predict(x_test)
    print 'baye gaussian:', round(gaussian.score(x_train, y_train)*100, 2)

    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)
    y_pred = perceptron.predict(x_test)
    print 'perceptron:', round(perceptron.score(x_train, y_train)*100, 2)

    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    y_pred = linear_svc.predict(x_test)
    print 'linear_svc: ', round(linear_svc.score(x_train, y_train)*100, 2)

    sgd = SGDClassifier()
    sgd.fit(x_train, y_train)
    y_pred = sgd.predict(x_test)
    print 'sgd: ', round(sgd.score(x_train, y_train)*100, 2)

    random_tree = RandomForestClassifier()
    random_tree.fit(x_train, y_train)
    y_pred = random_tree.predict(x_test)
    print 'random_tree: ', round(random_tree.score(x_train, y_train) * 100, 2)
    
    '''

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)
    print 'decision_tree: ', round(decision_tree.score(x_train, y_train)*100, 2)

    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred})
    #print submission
    submission.to_csv('./submission.csv', index=False)

    xgb = XGBClassifier()
    xgb.fit(x_train,y_train)
    y_pred = xgb.predict(x_test)
    print 'xgb:', round(xgb.score(x_train, y_train)*100, 2)


def test():

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print train_df['Survived']
    print train_df.describe()


if __name__ == '__main__':
    model()












