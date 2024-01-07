import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score,precision_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt
import re
import itertools
from sklearn.dummy import DummyClassifier
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def convert_doc_embedding(dataframe):
    # making sure the doc_embedding columns maintains the same format in the created and saved df
    def convert_to_list(string_repr):
        # extract the numeric values from the string
        numeric_values = [float(x) for x in re.findall(r'-?\d+\.\d+', string_repr)]
        return numeric_values
    dataframe['doc_embedding'] = dataframe['doc_embedding'].apply(convert_to_list)
    return dataframe


def select_20_features(df):
    # trains a logreg model on the data and returns the 20 features with the highest importance in a dataframe
    X_embeddings = np.vstack(df['doc_embedding'].to_numpy())
    X_doc_length = df['doc_embedding_average'].to_numpy().reshape(-1, 1)
    X_additional_features = df[['doc_length', 'nr_sent', 'avg_sentence_length',
                                '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT',
                                'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
                                'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                                'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
                                'WRB', '``', 'count_a', 'count_and', 'count_you', 'count_is', 'count_?',
                                'count_"', 'count_/', 'count_#', 'count_!']].to_numpy()

    # concatenate embeddings and additional features
    X = np.hstack([X_embeddings, X_doc_length, X_additional_features])
    y = df['binary_birth_year']

    # train a logistic regression on this
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # get coefficients and feature names
    coefficients = lr_model.coef_[0]
    feature_names_embeddings = [f'embedding_{i}' for i in range(X_embeddings.shape[1])]
    feature_names_additional = ['doc_embedding_average'] + [
        'doc_length', 'nr_sent', 'avg_sentence_length',
        '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT',
        'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
        'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
        'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
        'WRB', '``', 'count_a', 'count_and', 'count_you', 'count_is', 'count_?',
        'count_"', 'count_/', 'count_#', 'count_!'
    ]
    feature_names = feature_names_embeddings + feature_names_additional
    coefficients_dict = dict(zip(feature_names, coefficients))
    coefficients_dict_sorted = dict(sorted(coefficients_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    # 20 most important coefficients
    top_20_features = dict(itertools.islice(coefficients_dict_sorted.items(), 20))
    feature_keys = list(top_20_features.keys())
    feature_keys.append('binary_birth_year')
    df_selected_features = mil_and_genz_merged[feature_keys]
    return df_selected_features

def majority_baseline(df):
    X = df.drop(['binary_birth_year'], axis=1)
    y = df['binary_birth_year']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), confusion_matrix(y_test, y_pred)

def default_baseline(df):
    X = df.drop(['binary_birth_year'], axis=1)
    y = df['binary_birth_year']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), confusion_matrix(y_test, y_pred)

def logreg_gs(df):
    X = df.drop(['binary_birth_year'], axis=1)
    y = df['binary_birth_year']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = LogisticRegression(solver='liblinear')
    grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}
    grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring='accuracy')
    grid_clf_acc.fit(X_train, y_train)
    y_pred = grid_clf_acc.predict(X_test)
    results = pd.DataFrame(grid_clf_acc.cv_results_)

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), confusion_matrix(y_test, y_pred), grid_clf_acc.best_estimator_.get_params()['C'], grid_clf_acc.best_estimator_.get_params()['penalty'], results

def naive_bayes_gs(df):
    X = df.drop(['binary_birth_year'], axis=1)
    y = df['binary_birth_year']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = GaussianNB()
    grid_values = {'var_smoothing': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]}
    grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring='accuracy')
    grid_clf_acc.fit(X_train, y_train)
    y_pred = grid_clf_acc.predict(X_test)
    results = grid_clf_acc.cv_results_
    param_values = results['param_var_smoothing'].data
    mean_test_scores = results['mean_test_score']

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
        y_test, y_pred), confusion_matrix(y_test, y_pred), grid_clf_acc.best_estimator_.get_params()['var_smoothing'], param_values, mean_test_scores

def SVM_gs(df):
    X = df.drop(['binary_birth_year'], axis=1)
    y = df['binary_birth_year']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = SVC()
    grid_values = {'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'sigmoid']}
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, scoring = 'accuracy')
    grid_clf_acc.fit(X_train, y_train)
    y_pred = grid_clf_acc.predict(X_test)
    results = grid_clf_acc.cv_results_
    param_C_values = results['param_C'].data
    param_gamma_values = results['param_gamma'].data
    param_kernel_values = results['param_kernel'].data
    mean_test_scores = results['mean_test_score']

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
        y_test, y_pred), confusion_matrix(y_test, y_pred), grid_clf_acc.best_estimator_.get_params()['C'], grid_clf_acc.best_estimator_.get_params()['gamma'], grid_clf_acc.best_estimator_.get_params()['kernel'] ,param_C_values, param_gamma_values, param_kernel_values, mean_test_scores

# merge the dataframes
mil_and_genz = pd.read_csv('data/mil_and_genz.csv')
mil_and_genz = convert_doc_embedding(mil_and_genz)
mil_and_genz2 = pd.read_csv('data/mil_and_genz2.csv')
mil_and_genz_merged = pd.merge(mil_and_genz, mil_and_genz2, on=['auhtor_ID', 'post', 'birth_year', 'binary_birth_year', 'post_tokenized'])
mil_and_genz_merged.to_csv('data/mil_and_genz_merged.csv')

# select the most important features
df_selected_features = select_20_features(mil_and_genz_merged)

# majority baseline model
accuracy_m, precision_m, recall_m, f1_m, confusion_matrix_m = majority_baseline(df_selected_features)
print('Accuracy Score : ' + str(accuracy_m))
print('Precision Score : ' + str(precision_m))
print('Recall Score : ' + str(recall_m))
print('F1 Score : ' + str(f1_m))
print('Confusion Matrix : \n' + str(confusion_matrix_m))

# default logreg baseline model
accuracy_d, precision_d, recall_d, f1_d, confusion_matrix_d = default_baseline(df_selected_features)
print('Accuracy Score : ' + str(accuracy_d))
print('Precision Score : ' + str(precision_d))
print('Recall Score : ' + str(recall_d))
print('F1 Score : ' + str(f1_d))
print('Confusion Matrix : \n' + str(confusion_matrix_d))

# logreg model with gridsearch
accuracy_l, precision_l, recall_l, f1_l, confusion_matrix_l, C_l, penalty_l, results_l = logreg_gs(df_selected_features)
print('Gridsearch results:\n C: ' + str(C_l) + ' penalty: ' + str(penalty_l))
print('Accuracy Score : ' + str(accuracy_l))
print('Precision Score : ' + str(precision_l))
print('Recall Score : ' + str(recall_l))
print('F1 Score : ' + str(f1_l))
print('Confusion Matrix : \n' + str(confusion_matrix_l))

heatmap_data_l = results_l.pivot(index='param_C', columns='param_penalty', values='mean_test_score')
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data_l, annot=True, fmt='.3f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Logistic Regression Grid Search Results')
plt.xlabel('Penalty')
plt.ylabel('C')
plt.show()

# naive bayes with gridsearch
accuracy_n, precision_n, recall_n, f1_n, confusion_matrix_n, var_smoothing, param_values_n, mean_test_scores_n = naive_bayes_gs(df_selected_features)
print('Gridsearch results:\n var_smoothing: ' + str(var_smoothing))
print('Accuracy Score : ' + str(accuracy_l))
print('Precision Score : ' + str(precision_l))
print('Recall Score : ' + str(recall_l))
print('F1 Score : ' + str(f1_l))
print('Confusion Matrix : \n' + str(confusion_matrix_l))

heatmap_data_n = pd.DataFrame({'var_smoothing': param_values_n, 'Mean Test Score': mean_test_scores_n})
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data_n.set_index('var_smoothing'), annot=True, fmt='.3f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Naive Bayes Grid Search Results')
plt.xlabel('var_smoothing')
plt.ylabel('Mean Test Score')
plt.show()

# SVM with gridsearch
accuracy_s, precision_s, recall_s, f1_s, confusion_matrix_s, C_s, gamma_s, kernel_s, param_C_values, param_gamma_values, param_kernel_values, mean_test_scores = SVM_gs(df_selected_features)
print('Gridsearch results:\n C: ' + str(C_s) + ' gamma: ' + str(gamma_s) + ' kernel: ' + str(kernel_s))
print('Accuracy Score : ' + str(accuracy_s))
print('Precision Score : ' + str(precision_s))
print('Recall Score : ' + str(recall_s))
print('F1 Score : ' + str(f1_s))
print('Confusion Matrix : \n' + str(confusion_matrix_s))

heatmap_data_s = pd.DataFrame({'C': param_C_values, 'gamma': param_gamma_values, 'kernel': param_kernel_values, 'Mean Test Score': mean_test_scores})
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data_s.pivot_table(index=['gamma', 'kernel'], columns='C', values='Mean Test Score'),
            annot=True, fmt='.3f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Grid Search Results')
plt.xlabel('C')
plt.ylabel('gamma and kernel')
plt.show()

