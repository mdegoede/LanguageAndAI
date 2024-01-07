from vectorization_functions import Vectorizer
import pandas as pd
import ast
from sklearn.feature_extraction import DictVectorizer

data_processor = Vectorizer('data/birth_year.csv')
mil_and_genz = data_processor.mil_and_genz
data_processor.docu_length()
data_processor.count_sent()
data_processor.add_avg_sentence_length_column()
for item in ['a', 'and', 'you', 'is', '?', '"', '/', '#', '!']:
    data_processor.add_word_occurrences_column(item)
data_processor.add_pos_tags_column()
# als het niet werkt, uncomment deze:
#mil_and_genz.to_csv('data/mil_and_genz2.csv', index=False)
#mil_and_genz = pd.read_csv('data/mil_and_genz2.csv')
#mil_and_genz['pos_tags'] = mil_and_genz['pos_tags'].apply(ast.literal_eval)
vec = DictVectorizer(sparse=False)
pos_tags_features = pd.DataFrame(vec.fit_transform(mil_and_genz['pos_tags']), columns=vec.get_feature_names())
mil_and_genz = pd.concat([mil_and_genz, pos_tags_features], axis=1)
mil_and_genz.to_csv('data/mil_and_genz2.csv', index=False)
print(mil_and_genz, mil_and_genz.columns)