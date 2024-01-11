from vectorization_functions import Vectorizer
import pandas as pd
import ast
from sklearn.feature_extraction import DictVectorizer
from embeddings_functions import FasttextEmbedding
import fasttext
from classification_functions import ModelEvaluator
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_processor = Vectorizer('data/birth_year.csv')
mil_and_genz = data_processor.mil_and_genz
data_processor.docu_length()
data_processor.count_sent()
data_processor.add_avg_sentence_length_column()
for item in ['a', 'and', 'you', 'is', '?', '"', '/', '#', '!']:
    data_processor.add_word_occurrences_column(item)
data_processor.add_pos_tags_column()
vec = DictVectorizer(sparse=False)
pos_tags_features = pd.DataFrame(vec.fit_transform(mil_and_genz['pos_tags']), columns=vec.get_feature_names())
mil_and_genz = pd.concat([mil_and_genz, pos_tags_features], axis=1)
mil_and_genz.to_csv('data/mil_and_genz2.csv', index=False)
print(mil_and_genz, mil_and_genz.columns)

processor = FasttextEmbedding('data/birth_year.csv', 'result/trained_fasttext_embeddings.bin')
model = processor.train_fasttext_model()
# model = fasttext.load_model("result/trained_fasttext_embeddings.bin")
processor.process_and_add_embedding()
processor.calculate_and_add_average_column()
mil_and_genz = processor.mil_and_genz
mil_and_genz.to_csv('data/mil_and_genz.csv', index=False)
print(mil_and_genz, mil_and_genz.columns)

def convert_doc_embedding(dataframe):
    # making sure the doc_embedding columns maintains the same format in the created and saved df
    def convert_to_list(string_repr):
        # extract the numeric values from the string
        numeric_values = [float(x) for x in re.findall(r'-?\d+\.\d+', string_repr)]
        return numeric_values
    dataframe['doc_embedding'] = dataframe['doc_embedding'].apply(convert_to_list)
    return dataframe

# merge the embedding and vectorized dataframes
mil_and_genz = pd.read_csv('data/mil_and_genz.csv')
mil_and_genz = convert_doc_embedding(mil_and_genz)
mil_and_genz2 = pd.read_csv('data/mil_and_genz2.csv')
mil_and_genz_merged = pd.merge(mil_and_genz, mil_and_genz2, on=['auhtor_ID', 'post', 'birth_year', 'binary_birth_year', 'post_tokenized'])
mil_and_genz_merged.to_csv('data/mil_and_genz_merged.csv')
print(mil_and_genz_merged)

# Initialize ModelEvaluator with selected features
evaluator = ModelEvaluator(mil_and_genz_merged)
df_selected_features = evaluator.select_20_features()

# Update ModelEvaluator with selected features
evaluator.df_selected_features = df_selected_features

# majority baseline model
print('majority baseline model')
accuracy_m, precision_m, recall_m, f1_m, confusion_matrix_m = evaluator.majority_baseline()
print('Accuracy Score : ' + str(accuracy_m))
print('Precision Score : ' + str(precision_m))
print('Recall Score : ' + str(recall_m))
print('F1 Score : ' + str(f1_m))
print('Confusion Matrix : \n' + str(confusion_matrix_m))

# default logreg baseline model
print('default logreg baseline model')
accuracy_d, precision_d, recall_d, f1_d, confusion_matrix_d = evaluator.default_baseline()
print('Accuracy Score : ' + str(accuracy_d))
print('Precision Score : ' + str(precision_d))
print('Recall Score : ' + str(recall_d))
print('F1 Score : ' + str(f1_d))
print('Confusion Matrix : \n' + str(confusion_matrix_d))

# logreg model with gridsearch
print('logreg model with gridsearch')
accuracy_l, precision_l, recall_l, f1_l, confusion_matrix_l, C_l, penalty_l, results_l = evaluator.logreg_gs()
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
print('naive bayes with gridsearch')
accuracy_n, precision_n, recall_n, f1_n, confusion_matrix_n, var_smoothing, param_values_n, mean_test_scores_n = evaluator.naive_bayes_gs()
print('Gridsearch results:\n var_smoothing: ' + str(var_smoothing))
print('Accuracy Score : ' + str(accuracy_n))
print('Precision Score : ' + str(precision_n))
print('Recall Score : ' + str(recall_n))
print('F1 Score : ' + str(f1_n))
print('Confusion Matrix : \n' + str(confusion_matrix_n))

heatmap_data_n = pd.DataFrame({'var_smoothing': param_values_n, 'Mean Test Score': mean_test_scores_n})
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data_n.set_index('var_smoothing'), annot=True, fmt='.3f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Naive Bayes Grid Search Results')
plt.xlabel('var_smoothing')
plt.ylabel('Mean Test Score')
plt.show()

# SVM with gridsearch
print('SVM with gridsearch')
accuracy_s, precision_s, recall_s, f1_s, confusion_matrix_s, C_s, gamma_s, kernel_s, param_C_values, param_gamma_values, param_kernel_values, mean_test_scores = evaluator.SVM_gs()
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



