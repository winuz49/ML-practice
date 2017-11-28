# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn import cross_validation
import operator
from sklearn.feature_selection import SelectKBest, f_classif


def get_type_cabin(cabine):
    cabine_search = re.search('\d+', cabine)
    if cabine_search:
        num = cabine_search.group(0)
        if np.float64(num) % 2 == 0:
            return '2'
        return '1'
    return '0'


def get_person(passenger):
    age, sex = passenger
    if age < 18:
        return 'child'
    elif sex == 'female':
        return 'female_adult'
    else:
        return 'male_adult'


def main():
    # missing : age Embarked(0,1,2 for CSQ -1 for nan) Cabin   need: survived of test
    train = pd.read_csv('./train.csv', dtype={"Age": np.float64})
    test = pd.read_csv('./test.csv', dtype={'Age': np.float64})
    # print train.describe()
    print train.head(5)
    # print train.info()

    print train.info()
    print test.info()

    target = train["Survived"].values

    full = pd.concat([train, test])
    print full.info()
    # print full.describe()

    full['surname'] = full['Name'].apply(lambda x: x.split(',')[0].lower())
    full['Title'] = full['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5,
                     "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3,
                     "Don": 8, "Dona": 8, "Lady": 9, "Countess": 9, "Jonkheer": 9, "Sir": 8, "Capt": 7, "Ms": 2}
    full['TitleCat'] = full.loc[:, 'Title'].map(title_mapping)
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
    full['FamilySize'] = pd.cut(full['FamilySize'], bins=[0, 1, 4, 20], labels=[0, 1, 2])

    full['NameLength'] = full['Name'].apply(lambda x: len(x))
    full['Embarked'] = pd.Categorical(full.Embarked).codes

    full['Fare'] = full['Fare'].fillna(8.05)

    full = pd.concat([full, pd.get_dummies(full['Sex'])], axis=1)

    full['CabinCat'] = pd.Categorical(full['Cabin'].fillna('0').apply(lambda x: x[0])).codes
    full['Cabin'] = full['Cabin'].fillna(' ')
    full['CabinType'] = full['Cabin'].map(get_type_cabin)

    full = pd.concat([full, pd.DataFrame(full[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])], axis=1)
    full = pd.concat([full, pd.get_dummies(full['person'])], axis=1)

    print full.info()
    # FEATURES BASED ON TICKET
    table_ticket = pd.DataFrame(full['Ticket'].value_counts())
    table_ticket.rename(columns={'Ticket': 'Ticket_Members'}, inplace=True)

    table_ticket['Ticket_perishing_women'] = full.Ticket[(full.female_adult == 1.0) & (full.Survived == 0.0) &
                                                         ((full.Parch > 0) | (full.SibSp > 0))].value_counts()
    table_ticket['Ticket_perishing_women'] = table_ticket['Ticket_perishing_women'].fillna(0)
    table_ticket['Ticket_perishing_women'][(table_ticket['Ticket_perishing_women'] > 0)] = 1.0

    table_ticket['Ticket_surviving_men'] = full.Ticket[(full.male_adult == 1.0) & (full.Survived == 1.0) &
                                                       ((full.Parch > 0) | (full.SibSp > 0))].value_counts()
    table_ticket['Ticket_surviving_men'] = table_ticket['Ticket_surviving_men'].fillna(0)
    table_ticket['Ticket_surviving_men'][(table_ticket['Ticket_surviving_men'] > 0)] = 1.0

    table_ticket['Ticket_Id'] = pd.Categorical(table_ticket.index).codes
    table_ticket['Ticket_Id'][(table_ticket['Ticket_Members'] < 3)] = -1

    table_ticket['Ticket_Members'] = pd.cut(table_ticket['Ticket_Members'], bins=[0, 1, 4, 20], labels=[0, 1, 2])

    full = pd.merge(full, table_ticket, left_on='Ticket', right_index=True, how='left', sort=False)

    # FEATURES BASED ON SURNAME
    table_surname = pd.DataFrame(full['surname'].value_counts())
    table_surname.rename(columns={'surname': 'Surname_Members'}, inplace=True)

    table_surname['Surname_perishing_women'] = full.surname[(full.female_adult == 1.0) & (full.Survived == 0.0) &
                                                            ((full.Parch > 0) | (full.SibSp > 0))].value_counts()
    table_surname['Surname_perishing_women'] = table_surname['Surname_perishing_women'].fillna(0)
    table_surname['Surname_perishing_women'][table_surname['Surname_perishing_women'] > 0] = 1.0

    table_surname['Surname_surviving_men'] = full.surname[(full.male_adult == 1.0) & (full.Survived == 1.0) &
                                                          ((full.Parch > 0) | (full.SibSp > 0))].value_counts()
    table_surname['Surname_surviving_men'] = table_surname['Surname_surviving_men'].fillna(0)
    table_surname['Surname_surviving_men'][table_surname['Surname_surviving_men'] > 0] = 1.0

    table_surname['Surname_Id'] = pd.Categorical(table_surname.index).codes
    table_surname['Surname_Id'][table_surname['Surname_Members'] < 3] = -1
    table_surname['Surname_Members'] = pd.cut(table_surname['Surname_Members'], bins=[0, 1, 4, 20], labels=[0, 1, 2])

    full = pd.merge(full, table_surname, left_on='surname', right_index=True, how='left', sort=False)

    # AGE PROCESSING
    classers = ['Fare', 'Parch', 'Pclass', 'SibSp', 'TitleCat', 'CabinCat', 'female', 'male',
                'Embarked', 'FamilySize', 'NameLength', 'Ticket_Members', 'Ticket_Id']
    etr = ExtraTreesRegressor(n_estimators=200)

    x_train = full[classers][full['Age'].notnull()]
    y_train = full['Age'][full['Age'].notnull()]
    x_test = full[classers][full['Age'].isnull()]
    etr.fit(x_train, np.ravel(y_train))
    age_preds = etr.predict(x_test)
    full['Age'][full['Age'].isnull()] = age_preds
    print full['Age']

    # Features
    features = ['female', 'male', 'Age', 'male_adult', 'female_adult', 'child', 'TitleCat', 'Pclass',
                'Pclass', 'Ticket_Id', 'NameLength', 'CabinType', 'CabinCat', 'SibSp', 'Parch',
                'Fare', 'Embarked', 'Surname_Members', 'Ticket_Members', 'FamilySize',
                'Ticket_perishing_women', 'Ticket_surviving_men',
                'Surname_perishing_women', 'Surname_surviving_men']

    train = full[0:891].copy()
    test = full[891:].copy()

    selector = SelectKBest(f_classif, k=len(features))
    selector.fit(train[features], target)
    scores = -np.log10(selector.pvalues_)
    indices = np.argsort(scores)[::-1]
    print ('Important Features:')
    for f in range(len(scores)):
        print("%0.2f %s" % (scores[indices[f]], features[indices[f]]))

    # BEST CLASSIFIER METHOD ==> RANDOM FOREST
    rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0: 0.745, 1: 0.255})

    # CROSS VALIDATION WITH RANDOM FOREST CLASSIFIER METHOD
    kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)
    scores = cross_validation.cross_val_score(rfc, train[features], target, cv=kf)
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean() * 100, scores.std() * 100, 'RFC Cross Validation'))
    rfc.fit(train[features], target)
    score = rfc.score(train[features], target)
    print("Accuracy: %0.3f            [%s]" % (score * 100, 'RFC full test'))
    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(features)):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f] + 1, importances[indices[f]] * 100, features[indices[f]]))

    # PREDICTION
    rfc.fit(train[features], target)
    predictions = rfc.predict(test[features])

    # OUTPUT FILE
    PassengerId = np.array(test["PassengerId"]).astype(int)
    my_prediction = pd.DataFrame(predictions, PassengerId, columns=["Survived"])

    my_prediction.to_csv("submission.csv", index_label=["PassengerId"])

    print 'finish'


if __name__ == '__main__':
    main()




