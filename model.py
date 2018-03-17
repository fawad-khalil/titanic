# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

#Get the training set (CSV file)
training_df = pandas.read_csv('D:/ML work/Titanic Data/train.csv', header=None)

#Assign first row as column headers
training_df.columns = training_df.iloc[0]

#Remove first row
training_df.drop(training_df.index[0], inplace = True)

#Convert columns to appropriate type
training_df['Age'] = training_df['Age'].astype(np.float64)
training_df['Survived'] = training_df['Survived'].astype(np.int16)
training_df['Fare'] = training_df['Fare'].astype(np.float64)

#Get median of ages
age_median = training_df['Age'].median()

#Replace nan values with median of ages of people (median is more robust to outliers than mean) 
training_df['Age'].fillna(age_median, inplace=True)

#print(training_df.count())

#survived_sex = training_df[training_df['Survived'] == 1]['Sex'].value_counts()

survived_sex_count = training_df.loc[training_df.Survived == 1, 'Sex'].value_counts()

nonsurvived_sex_count = training_df.loc[training_df.Survived == 0, 'Sex'].value_counts()

#survival_df = pandas.DataFrame([survived_passanger_count, nonsurvived_passanger_count])
#survival_df.index = ['Survived', 'Non-Survived']
#
#survival_df.plot(kind = 'bar', stacked = True, figsize = (20, 12))

survived_count = training_df.loc[training_df.Survived == 1, 'Age']
nonsurvived_count = training_df.loc[training_df.Survived == 0, 'Age']

survival_df = pandas.DataFrame([survived_count, nonsurvived_count])

plt.figure(figsize=(20, 12))

##Plot stacked histogram based on ages and survived/nonsurvived
#plt.hist([training_df.loc[training_df.Survived == 1, 'Age'], training_df.loc[training_df.Survived == 0, 'Age']], color = ['g', 'r'], stacked = True, bins = 30, label = ['Survived', 'Non-Survived'])
#plt.xlabel('Age')
#plt.ylabel('No. of passanger')
#
#plt.legend()
##End Plot

##Plot stacked histogram on the basis of fare and survived/non-survived
#plt.hist([training_df.loc[training_df.Survived == 1, 'Fare'], training_df.loc[training_df.Survived == 0, 'Fare']], color = ['g', 'r'], stacked = True, bins = 30, label = ['Survived', 'Non-Survived'])
#plt.xlabel('Fare')
#plt.ylabel('No. of passanger')
#
#plt.legend()
##End Plot

##Plot scatter plot to see correlation of survival with fare and Age 
#ax = plt.subplot()
#
#ax.scatter(training_df.loc[training_df.Survived == 1, 'Age'], training_df.loc[training_df.Survived == 1, 'Fare'], color = 'g', s = 40)
#ax.scatter(training_df.loc[training_df.Survived == 0, 'Age'], training_df.loc[training_df.Survived == 0, 'Fare'], color = 'r', s = 40)
#
#ax.set_xlabel('Age')
#ax.set_ylabel('Fare')
#ax.legend(('Survived', 'Non-Survived'), scatterpoints = 1, loc = 'upper right', fontsize = 15)
##End plot

##Plot stacked bar graph to see correlation with pclass and average ticket fare
#ax = plt.subplot()
#
#ax.set_ylabel('Average Fare')
#training_df.groupby('Pclass').mean().Fare.plot(kind = 'bar', figsize = (20, 12), ax = ax)
##End plot

##Plot the bar graph based on survived/non-survived and embarkion site
#survived_count = training_df.loc[training_df.Survived == 1, 'Embarked'].value_counts()
#nonsurvived_count = training_df.loc[training_df.Survived == 0, 'Embarked'].value_counts()
#
#survival_df = pandas.DataFrame([survived_count, nonsurvived_count])
#survival_df.index = ['Survived', 'Non-Survived']
#survival_df.plot(kind = 'bar', color = ['blue', 'green', 'red'], stacked = True)
##End plot


#placeholder is the name of row or column for which dummies are required
def replace_with_dummies(df, placeholder, axis_flag):
    dummies = pandas.get_dummies(df[placeholder], prefix = placeholder)
    df = pandas.concat([df, dummies], axis = axis_flag)
    
    df.drop(placeholder, axis = axis_flag, inplace = True)
    
    return df
#End replace_with_dummies

def parse_ticket(ticket):
    ticket = ticket.replace('.', '').replace('/', '')
    ticket = ticket.split()
    
    ticket = map(lambda x: x.strip(), ticket)
    
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    
    if len(ticket) > 0:
        return ticket[0]
    
    return 'XXX'
#end parse_ticket()  

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
#end compute_score  

#Now we combine training and test dataset to perform some feature engineering
training_df_2 = pandas.read_csv('D:/ML work/Titanic Data/train.csv', header=None)

test_df = pandas.read_csv('D:/ML work/Titanic Data/test.csv', header=None)

#Assign first row as column headers
training_df_2.columns = training_df_2.iloc[0]
test_df.columns = test_df.iloc[0]

#Remove first row
training_df_2.drop(training_df_2.index[0], inplace = True)
test_df.drop(test_df.index[0], inplace = True)

combined_df = training_df_2.append(test_df)

combined_df.reset_index(inplace = True)

combined_df.drop('index', axis = 1, inplace = True)

survived_column = combined_df.Survived.iloc[ : 891]
survived_column.astype(np.int16)

combined_df.drop('Survived', axis = 1, inplace = True)

combined_df['Title'] = combined_df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

#Dictionary of titles
title_dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
combined_df.Title.map(title_dictionary)

combined_df['Age'] = combined_df['Age'].astype(np.float32)
combined_df['SibSp'] = combined_df['SibSp'].astype(np.float16)
combined_df['Fare'] = combined_df['Fare'].astype(np.float64)
combined_df['Parch'] = combined_df['Parch'].astype(np.float16)

grouped_train_median = combined_df.iloc[ : 891].groupby(['Sex', 'Title']).median()

grouped_test_median = combined_df.iloc[891 : ].groupby(['Sex', 'Title']).median()

#Training ages filling
group_median = combined_df.iloc[ : 891].groupby(['Sex', 'Pclass', 'Title'])['Age'].transform('median')

combined_df['Age'] = combined_df['Age'].fillna(group_median)

#Test ages filling
group_median = combined_df.iloc[891 : ].groupby(['Sex', 'Pclass', 'Title'])['Age'].transform('median')

combined_df['Age'] = combined_df['Age'].fillna(group_median)

combined_df.Age.iloc[ : 891].fillna(combined_df.iloc[: 891].Age.median(), inplace = True)
combined_df.Age.iloc[ 891 : ].fillna(combined_df.iloc[ 891 : ].Age.median(), inplace = True)

#we don't need names anymore as we've created titles
combined_df.drop('Name', axis = 1, inplace = True)

combined_df = replace_with_dummies(combined_df, 'Title', 1)

#titles_dummy = pandas.get_dummies(combined_df['Title'], prefix = 'Title')
#
#combined_df = pandas.concat([combined_df, titles_dummy], axis = 1)
#
##Drop titles because we now have encoding
#combined_df.drop('Title', axis = 1, inplace = True)

training_fare_median = combined_df.iloc[ : 891 ].Fare.mean()
test_fare_median = combined_df.iloc[ 891 : ].Fare.mean()

combined_df.Fare.iloc[ : 891].fillna(training_fare_median, inplace = True)
combined_df.Fare.iloc[ 891 : ].fillna(test_fare_median, inplace = True)

temp_df = combined_df.dropna(subset=['Embarked'])

combined_df.Embarked.fillna(temp_df['Embarked'].value_counts().idxmax())

embarked_dummies = pandas.get_dummies(combined_df['Embarked'], prefix = 'Embarked')

combined_df = pandas.concat([combined_df, embarked_dummies], axis = 1)
combined_df.drop('Embarked', axis = 1, inplace = True)

combined_df.Cabin.fillna('Uknown', inplace = True)

combined_df.Sex = combined_df.Sex.map({'male': 0, 'female': 1})

combined_df['Familysize'] = combined_df['Parch'] + combined_df['SibSp'] + 1

combined_df['Familysize'] = combined_df['Familysize'].map(lambda x: 'single' if x == 1 else 'small' if 1 < x < 4 else 'large' )

familysize_dummies = pandas.get_dummies(combined_df['Familysize'], prefix = 'Familysize')

combined_df = pandas.concat([combined_df, familysize_dummies], axis = 1)
combined_df.drop('Familysize', axis = 1, inplace = True)

combined_df['Ticket'] = combined_df.Ticket.map(parse_ticket)

ticket_dummies = pandas.get_dummies(combined_df.Ticket, prefix = 'Ticket')

combined_df = pandas.concat([combined_df, ticket_dummies], axis = 1)
combined_df.drop('Ticket', axis = 1, inplace = True)

pclass_dummies = pandas.get_dummies(combined_df.Pclass, prefix = 'Pclass')

combined_df = pandas.concat([combined_df, pclass_dummies], axis = 1)
combined_df.drop('Pclass', axis = 1, inplace = True)

cabin_dummies = pandas.get_dummies(combined_df.Cabin, prefix = 'Cabin')

combined_df = pandas.concat([combined_df, cabin_dummies], axis = 1)

combined_df.drop('Cabin', axis = 1, inplace = True)

combined_df.drop('PassengerId', axis = 1, inplace = True)

training_df_processed = combined_df.iloc[ : 891]

test_df_processed = combined_df.iloc[ 891 : ]

clf = clf.fit(training_df_processed, survived_column)

features = pandas.DataFrame()
features['feature'] = training_df_processed.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

#features.plot(kind='barh', figsize=(30, 90))

model = SelectFromModel(clf, prefit = True)

train_reduced = model.transform(training_df_processed)

#print(train_reduced.shape)
#print(test_df_processed.isnull().any().any())
test_reduced = model.transform(test_df_processed)

#print(test_reduced.shape)

parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
model = RandomForestClassifier(**parameters)
model.fit(training_df_processed, survived_column)

print(compute_score(model, training_df_processed, survived_column, scoring='accuracy'))

output = model.predict(test_df_processed).astype(int)

df_output = pandas.DataFrame()
aux = pandas.read_csv('D:/ML work/Titanic Data/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('D:/ML work/Titanic Data/output.csv',index=False)
