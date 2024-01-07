import pandas as pd
import fasttext
import numpy as np

class FasttextEmbedding:
    def __init__(self, data_path, output_model_path):
        self.df_birth_year = pd.read_csv(data_path)
        self.mil_and_genz = self.load_data()
        self.output_model_path = output_model_path
        self.model = None

    def load_data(self):
        mil_and_genz = self.df_birth_year[(1986 < self.df_birth_year['birth_year']) & (self.df_birth_year['birth_year'] <= 2006)]
        mil_and_genz['binary_birth_year'] = 1
        mil_and_genz.loc[
            (1996 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 2006), 'binary_birth_year'] = 0
        mil_and_genz = mil_and_genz.reset_index(drop=True)
        mil_and_genz['post_tokenized'] = mil_and_genz.post.str.findall('\w+|[^\w\s]')
        return mil_and_genz

    def train_fasttext_model(self):
        corpus = " ".join([" ".join(post) for post in self.mil_and_genz['post_tokenized']])
        with open("data/temp_corpus.txt", "w", encoding="utf-8") as file:
            file.write(corpus)
        self.model = fasttext.train_unsupervised("data/temp_corpus.txt")
        self.model.save_model(self.output_model_path)
        return self.model

    def process_and_add_embedding(self):
        if self.model is not None:
            def get_average_embedding(tokens):
                embeddings = [self.model.get_word_vector(word) for word in tokens if word in self.model.words]
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                else:
                    avg_embedding = np.zeros(self.model.get_dimension())
                return avg_embedding

            self.mil_and_genz['doc_embedding'] = self.mil_and_genz['post_tokenized'].apply(get_average_embedding)

    def calculate_and_add_average_column(self):
        def calculate_average(lst):
            return sum(lst) / len(lst) if len(lst) > 0 else None
        self.mil_and_genz['doc_embedding_average'] = self.mil_and_genz['doc_embedding'].apply(calculate_average)

