import pandas as pd
import nltk
from nltk import pos_tag
from sklearn.feature_extraction import DictVectorizer

class Vectorizer:
    def __init__(self, data_path):
        self.df_birth_year = pd.read_csv(data_path)
        self.mil_and_genz = self.load_data()

    def load_data(self):
        """
        loads the data and tokenize the text
        """
        mil_and_genz = self.df_birth_year[(1986 < self.df_birth_year['birth_year']) & (self.df_birth_year['birth_year'] <= 2006)]
        mil_and_genz['binary_birth_year'] = 1
        mil_and_genz.loc[
            (1996 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 2006), 'binary_birth_year'] = 0
        mil_and_genz = mil_and_genz.reset_index(drop=True)
        mil_and_genz['post_tokenized'] = mil_and_genz.post.str.findall('\w+|[^\w\s]')
        return mil_and_genz

    def docu_length(self):
        """
        calculates the document length of a post
        """
        def count_words(tokens):
            return sum(1 for token in tokens if token.isalpha())
        self.mil_and_genz['doc_length'] = self.mil_and_genz['post_tokenized'].apply(count_words)

    def count_sent(self):
        """
        calculates number of sentences per document
        """
        self.mil_and_genz['nr_sent'] = [len(nltk.sent_tokenize(text)) for text in self.mil_and_genz['post']]

    def add_avg_sentence_length_column(self):
        """
        calculate average sentence length and add the column to the dataframe
        """
        def calculate_avg_sentence_length(text):
            return sum(1 for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence) if word.isalpha()) / max(1, len(nltk.sent_tokenize(text)))
        self.mil_and_genz['avg_sentence_length'] = self.mil_and_genz['post'].apply(calculate_avg_sentence_length)

    def add_pos_tags_column(self):
        """
        create a dictionary of POS tags for every post, also adds POS as separate columns
        """
        def get_pos_tags(tokenized_post):
            pos_tags = pos_tag(tokenized_post)
            pos_counts = {}
            for _, tag in pos_tags:
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
            return pos_counts
        self.mil_and_genz['pos_tags'] = self.mil_and_genz['post_tokenized'].apply(get_pos_tags)
        vec = DictVectorizer(sparse=False)
        pos_tags_features = pd.DataFrame(vec.fit_transform(self.mil_and_genz['pos_tags']), columns=vec.get_feature_names())
        self.mil_and_genz = pd.concat([self.mil_and_genz, pos_tags_features], axis=1)

    def add_word_occurrences_column(self, word):
        """
        counts occurrences of a given word
        """
        def count_occurrences(row):
            tokens = [token.lower() for token in row['post_tokenized']]
            return tokens.count(word.lower())
        column_name = f'count_{word}'
        self.mil_and_genz[column_name] = self.mil_and_genz.apply(count_occurrences, axis=1)

