import pandas as pd
import nltk
from nltk import pos_tag
from sklearn.feature_extraction import DictVectorizer

def load_data(data):
    # loads the data and tokenize the text
    df_birth_year = pd.read_csv(data)
    mil_and_genz = df_birth_year[(1986 < df_birth_year['birth_year']) & (df_birth_year['birth_year'] <= 2006)]
    mil_and_genz['binary_birth_year'] = 1
    mil_and_genz.loc[
        (1996 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 2006), 'binary_birth_year'] = 0
    mil_and_genz = mil_and_genz.reset_index(drop=True)
    mil_and_genz['post_tokenized'] = mil_and_genz.post.str.findall('\w+|[^\w\s]')
    return mil_and_genz

def docu_length(dataframe):
    # calculates the document length of a post
    def count_words(tokens):
        return sum(1 for token in tokens if token.isalpha())
    dataframe['doc_length'] = dataframe['post_tokenized'].apply(count_words)
    return dataframe

def count_sent(dataframe):
    # calculates number of sentences per document
    dataframe['nr_sent'] = [len(nltk.sent_tokenize(text)) for text in dataframe['post']]
    return dataframe

def add_avg_sentence_length_column(dataframe):
    def calculate_avg_sentence_length(text):
        return sum(1 for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence) if word.isalpha()) / max(1, len(nltk.sent_tokenize(text)))
    dataframe['avg_sentence_length'] = dataframe['post'].apply(calculate_avg_sentence_length)
    return dataframe

def add_pos_tags_column(dataframe):
    # create a dictionary of POS tags for every post, also adds POS as separate columns
    def get_pos_tags(tokenized_post):
        pos_tags = pos_tag(tokenized_post)
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        return pos_counts
    dataframe['pos_tags'] = dataframe['post_tokenized'].apply(get_pos_tags)
    vec = DictVectorizer(sparse=False)
    pos_tags_features = pd.DataFrame(vec.fit_transform(dataframe['pos_tags']), columns=vec.get_feature_names())
    dataframe = pd.concat([dataframe, pos_tags_features], axis=1)
    return dataframe

def count_occurrences(row, word):
    # counts occurences of a given word
    tokens = [token.lower() for token in row['post_tokenized']]
    return tokens.count(word.lower())

mil_and_genz = load_data('data/birth_year.csv')
mil_and_genz = docu_length(mil_and_genz)
mil_and_genz = count_sent(mil_and_genz)
mil_and_genz = add_avg_sentence_length_column(mil_and_genz)
mil_and_genz = add_pos_tags_column(mil_and_genz)
mil_and_genz['count_a'] = mil_and_genz.apply(count_occurrences, args=('a',), axis=1)
mil_and_genz['count_and'] = mil_and_genz.apply(count_occurrences, args=('and',), axis=1)
mil_and_genz['count_you'] = mil_and_genz.apply(count_occurrences, args=('you',), axis=1)
mil_and_genz['count_is'] = mil_and_genz.apply(count_occurrences, args=('is',), axis=1)
mil_and_genz['count_?'] = mil_and_genz.apply(count_occurrences, args=('?',), axis=1)
mil_and_genz['count_"'] = mil_and_genz.apply(count_occurrences, args=('"',), axis=1)
mil_and_genz['count_/'] = mil_and_genz.apply(count_occurrences, args=('/',), axis=1)
mil_and_genz['count_#'] = mil_and_genz.apply(count_occurrences, args=('#',), axis=1)
mil_and_genz['count_!'] = mil_and_genz.apply(count_occurrences, args=('!',), axis=1)
print(mil_and_genz)
mil_and_genz.to_csv('data/mil_and_genz2.csv', index=False)

