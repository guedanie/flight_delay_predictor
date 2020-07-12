from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint 
import pandas as pd


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans

# import preprocessing_permits as pr
import numpy as np

############################################################################################################
#                                   Cross Validation Modeling                                              #
############################################################################################################

def run_decision_tree_cv(X_train, y_train):
    '''
    Function to run a decision tree model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''


    #  keys are names of hyperparams, values are a list of values to try for that hyper parameter
    params = {
        'max_depth': range(1, 11),
        'criterion': ['gini', 'entropy']
    }

    dtree = DecisionTreeClassifier()

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(dtree, params, cv=3, scoring= "recall")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def run_random_forest_cv(X_train, y_train):
    '''
    Function to run a random forest model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''
    
    params = {
    'max_depth': range(1, 10),
    "min_samples_leaf": range(1,10)
    }

    rf = RandomForestClassifier(random_state = 123)

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(rf, params, cv=3, scoring= "recall")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def run_knn_cv(X_train, y_train):
    '''
    Function to run a knn model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''

    knn = KNeighborsClassifier()

    params = {
        'weights': ["uniform", "distance"],
        "n_neighbors": range(1,20)
    }

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(knn, params, cv=3, scoring= "recall")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def evaluate_on_test_data(X_test, y_test, model):
    model.score(X_test, y_test)

def create_prediction(df, model):
    y_pred = model.predict(df)
    return y_pred


############################################################################################################
#                                        Evaluations                                                       #
############################################################################################################

def create_report(y_train, y_pred):
    '''
    Helper function used to create a classification evaluation report, and return it as df
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    return report


def accuracy_report(model, y_pred, y_train):
    '''
    Main function used to create printable versions of the classification accuracy score, confusion matrix and classification report.
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    accuracy_score = f'Accuracy on dataset: {report.accuracy[0]:.2f}'

    labels = sorted(y_train.unique())
    matrix = pd.DataFrame(confusion_matrix(y_train, y_pred), index = labels, columns = labels)

    return accuracy_score, matrix, report


############################################################################################################
#                                        Traditional Modeling                                              #
############################################################################################################

# ---------------------- #
#        Models          #
# ---------------------- #

# Decision Tree

def run_clf(X_train, y_train, max_depth):
    '''
    Function used to create and fit decision tree models. It requires a max_depth parameter. Returns model and predictions.
    '''
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return clf, y_pred

def run_clf_loop(train_scaled, validate_scaled, y_validate, y_train, max_range):
    '''
    Function used to run through a loop, comparing each score at the different hyperparameters, to understand what is the best hyperparameter to use. Takes the scaled train and validate dfs, as well as the target train and validate df. Also takes a max range, for. the loop
    '''
    for i in range(1, max_range):
        clf, y_pred = run_clf(train_scaled, y_train, i)
        score = clf.score(train_scaled, y_train)
        validate_score = clf.score(validate_scaled, y_validate)
        _, _, report = accuracy_report(clf, y_pred, y_train)
        recall_score = report["True"].recall
        print(f"Max_depth = {i}, accuracy_score = {score:.2f}. validate_score = {validate_score:.2f}, recall = {recall_score:.2f}")

def clf_feature_importances(clf, train_scaled):
    '''
    Function used to create a graph, which ranks the features based on which were more important for the modeling
    '''
    coef = clf.feature_importances_
    # We want to check that the coef array has the same number of items as there are features in our X_train dataframe.
    assert(len(coef) == train_scaled.shape[1])
    coef = clf.feature_importances_
    columns = train_scaled.columns
    df = pd.DataFrame({"feature": columns,
                    "feature_importance": coef,
                    })

    df = df.sort_values(by="feature_importance", ascending=False)
    sns.barplot(data=df, x="feature_importance", y="feature", palette="Blues_d")
    plt.title("What are the most influencial features?")

# KNN

def run_knn(X_train, y_train, n_neighbors):
    '''
    Function used to create and fit KNN model. Requires to specify n_neighbors. Returns model and predictions.
    '''
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    return knn, y_pred

def run_knn_loop(train_scaled, validate_scaled, y_validate, y_train, max_range):
    '''
    Function used to run through a loop, comparing each score at the different hyperparameters, to understand what is the best hyperparameter to use. Takes the scaled train and validate dfs, as well as the target train and validate df. Also takes a max range, for. the loop
    '''
    for i in range(1, max_range):
        knn, y_pred = run_knn(train_scaled, y_train, i)
        score = knn.score(train_scaled, y_train)
        validate_score = knn.score(validate_scaled, y_validate)
        _, _, report = accuracy_report(knn, y_pred, y_train)
        recall_score = report["True"].recall
        print(f"k_n = {i}, accuracy_score = {score:.2f}. validate_score = {validate_score:.2f}, recall = {recall_score:.2f}")


# Random_forest

def run_rf(X_train, y_train, leaf, max_depth):
    ''' 
    Function used to create and fit random forest models. Requires to specif leaf and max_depth. Returns model and predictions.
    '''
    rf = RandomForestClassifier(random_state= 123, min_samples_leaf = leaf, max_depth = max_depth).fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred

def run_rf_loop(train_scaled, validate_scaled, y_validate, y_train, max_range):
    '''
    Function used to run through a loop, comparing each score at the different hyperparameters, to understand what is the best hyperparameter to use. Takes the scaled train and validate dfs, as well as the target train and validate df. Also takes a max range, for. the loop
    '''
    for i in range(1, max_range):
        rf, y_pred = run_rf(train_scaled, y_train, 1, i)
        score = rf.score(train_scaled, y_train)
        validate_score = rf.score(validate_scaled, y_validate)
        _, _, report = accuracy_report(rf, y_pred, y_train)
        recall_score = report["True"].recall
        print(f"Max_depth = {i}, accuracy_score = {score:.2f}. validate_score = {validate_score:.2f}, recall = {recall_score:.2f}")

def rf_feature_importance(rf, train_scaled):
    '''
    Function used to create a graph, which ranks the features based on which were more important for the modeling
    '''
    coef = rf.feature_importances_
    columns = train_scaled.columns
    df = pd.DataFrame({"feature": columns,
                    "feature_importance": coef,
                    })

    df = df.sort_values(by="feature_importance", ascending=False)
    sns.barplot(data=df, x="feature_importance", y="feature", palette="Blues_d")
    plt.title("What are the most influencial features?")

