import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score,precision_score,f1_score, confusion_matrix
import itertools
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class ModelEvaluator:
    def __init__(self, mil_and_genz_merged):
        self.mil_and_genz_merged = mil_and_genz_merged
        self.df_selected_features = None

    def select_20_features(self):
        """
        trains a logreg model on the data and returns the 20 features with the highest importance in a dataframe
        """
        X_embeddings = np.vstack(self.mil_and_genz_merged['doc_embedding'].to_numpy())
        X_doc_length = self.mil_and_genz_merged['doc_embedding_average'].to_numpy().reshape(-1, 1)
        X_additional_features = self.mil_and_genz_merged[['doc_length', 'nr_sent', 'avg_sentence_length',
                                    '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT',
                                    'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
                                    'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                                    'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
                                    'WRB', '``', 'count_a', 'count_and', 'count_you', 'count_is', 'count_?',
                                    'count_"', 'count_/', 'count_#', 'count_!']].to_numpy()

        # concatenate embeddings and additional features
        X = np.hstack([X_embeddings, X_doc_length, X_additional_features])
        y = self.mil_and_genz_merged['binary_birth_year']

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
        df_selected_features = self.mil_and_genz_merged[feature_keys]
        return df_selected_features

    def majority_baseline(self):
        """
        majority baseline model
        """
        X = self.df_selected_features.drop(['binary_birth_year'], axis=1)
        y = self.df_selected_features['binary_birth_year']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        clf = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
            y_test, y_pred), confusion_matrix(y_test, y_pred)

    def default_baseline(self):
        """
        default logreg baseline model
        """
        X = self.df_selected_features.drop(['binary_birth_year'], axis=1)
        y = self.df_selected_features['binary_birth_year']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        clf = LogisticRegression().fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
            y_test, y_pred), confusion_matrix(y_test, y_pred)

    def logreg_gs(self):
        """
        logreg model with gridsearch
        """
        X = self.df_selected_features.drop(['binary_birth_year'], axis=1)
        y = self.df_selected_features['binary_birth_year']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        clf = LogisticRegression(solver='liblinear')
        grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}
        grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring='accuracy')
        grid_clf_acc.fit(X_train, y_train)
        y_pred = grid_clf_acc.predict(X_test)
        results = pd.DataFrame(grid_clf_acc.cv_results_)

        return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
            y_test, y_pred), confusion_matrix(y_test, y_pred), grid_clf_acc.best_estimator_.get_params()['C'], \
               grid_clf_acc.best_estimator_.get_params()['penalty'], results

    def naive_bayes_gs(self):
        """
        naive bayes with gridsearch
        """
        X = self.df_selected_features.drop(['binary_birth_year'], axis=1)
        y = self.df_selected_features['binary_birth_year']
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
            y_test, y_pred), confusion_matrix(y_test, y_pred), grid_clf_acc.best_estimator_.get_params()[
                   'var_smoothing'], param_values, mean_test_scores

    def SVM_gs(self):
        """
        SVM with gridsearch
        """
        X = self.df_selected_features.drop(['binary_birth_year'], axis=1)
        y = self.df_selected_features['binary_birth_year']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        clf = SVC()
        grid_values = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                       'kernel': ['rbf', 'sigmoid']}
        grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring='accuracy')
        grid_clf_acc.fit(X_train, y_train)
        y_pred = grid_clf_acc.predict(X_test)
        results = grid_clf_acc.cv_results_
        param_C_values = results['param_C'].data
        param_gamma_values = results['param_gamma'].data
        param_kernel_values = results['param_kernel'].data
        mean_test_scores = results['mean_test_score']

        return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(
            y_test, y_pred), confusion_matrix(y_test, y_pred), grid_clf_acc.best_estimator_.get_params()['C'], \
               grid_clf_acc.best_estimator_.get_params()['gamma'], grid_clf_acc.best_estimator_.get_params()[
                   'kernel'], param_C_values, param_gamma_values, param_kernel_values, mean_test_scores

