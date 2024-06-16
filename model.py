import os
import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import seaborn as sns
import numpy as np

# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
sns.set_style("whitegrid", {'axes.grid': False})
pio.templates.default = "plotly_white"

def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\n')
    print('Dataset columns:', df.columns)
    print('\n')
    print('Data types of each column: ')
    print(df.dtypes)

def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')

def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def GetModel():
    Models = []
    Models.append(('LR', LogisticRegression()))
    Models.append(('LDA', LinearDiscriminantAnalysis()))
    Models.append(('KNN', KNeighborsClassifier()))
    Models.append(('CART', DecisionTreeClassifier()))
    Models.append(('NB', GaussianNB()))
    Models.append(('SVM', SVC(probability=True)))
    return Models

def ensemblemodels():
    ensembles = []
    ensembles.append(('AB', AdaBoostClassifier()))
    ensembles.append(('GBM', GradientBoostingClassifier()))
    ensembles.append(('RF', RandomForestClassifier()))
    ensembles.append(('Bagging', BaggingClassifier()))
    ensembles.append(('ET', ExtraTreesClassifier()))
    return ensembles

def NormalizedModel(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler == 'minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'normalizer':
        scaler = Normalizer()
    elif nameOfScaler == 'binarizer':
        scaler = Binarizer()

    pipelines = [(nameOfScaler + 'LR', Pipeline([('Scaler', scaler), ('LR', LogisticRegression())])),
                 (nameOfScaler + 'LDA', Pipeline([('Scaler', scaler), ('LDA', LinearDiscriminantAnalysis())])),
                 (nameOfScaler + 'KNN', Pipeline([('Scaler', scaler), ('KNN', KNeighborsClassifier())])),
                 (nameOfScaler + 'CART', Pipeline([('Scaler', scaler), ('CART', DecisionTreeClassifier())])),
                 (nameOfScaler + 'NB', Pipeline([('Scaler', scaler), ('NB', GaussianNB())])),
                 (nameOfScaler + 'SVM', Pipeline([('Scaler', scaler), ('SVM', SVC())])),
                 (nameOfScaler + 'AB', Pipeline([('Scaler', scaler), ('AB', AdaBoostClassifier())])),
                 (nameOfScaler + 'GBM', Pipeline([('Scaler', scaler), ('GMB', GradientBoostingClassifier())])),
                 (nameOfScaler + 'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier())])),
                 (nameOfScaler + 'ET', Pipeline([('Scaler', scaler), ('ET', ExtraTreesClassifier())]))]

    return pipelines

def fit_model(X_train, y_train, models):
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return names, results

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

'''def classification_metrics(model, X_train, y_train, X_test, y_test):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))'''

# Load Dataset
df = pd.read_csv('Crop_recommendation.csv')

# Explore Data
explore_data(df)
checking_removing_duplicates(df)


# Split Data to Training and Validation set
target = 'label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

# Train model
pipeline = make_pipeline(StandardScaler(), GaussianNB())
model = pipeline.fit(X_train, y_train)
classification_metrics(model, X_train, y_train, X_test, y_test)

# Save the model
model_filename = 'model.pkl'
model_dir = os.path.dirname(os.path.abspath("/content/sample_data"))
model_path = os.path.join(model_dir, model_filename)
save_model(model, model_path)

print("Model saved successfully.")
