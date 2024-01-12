import pandas as pd
import fasttext
import numpy as np
import os
import langdetect
from langdetect import detect, detect_langs

class FasttextEmbedding:
    def __init__(self, data_path, output_model_path):
        self.df_birth_year = pd.read_csv(data_path)
        self.mil_and_genz = self.load_data()
        self.mil_and_genz = self.language_detector_non_eng()
        self.output_model_path = output_model_path
        self.model = None

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

    def language_detection(self, text, method="single"):

        """
        @desc:
          - detects the language of a text
        @params:
          - text: the text which language needs to be detected
          - method: detection method:
            single: if the detection is based on the first option (detect)
        @return:
          - the langue/list of languages
        """

        if (method.lower() != "single"):
            result = detect_langs(text)

        else:
            result = detect(text)

        return result

    def language_detector_non_eng(self):
        """Takes the data as input and returns the data without non-English entries or below 85% of English."""
        print('Detecting non english and dropping...')


        dropping_no_eng = list()

        for i in range(len(self.mil_and_genz)):
            try:
                x = self.language_detection(str(self.mil_and_genz['post'][i]), 'all languages')
                if len(x) > 1:
                    #print(f"index: {i}, score: {x}")
                    # Remove brackets and split the string by ','
                    pairs = str(x).strip('[]').split(', ')

                    # Create a dictionary for each key-value pair
                    result_dict = {}
                    for pair in pairs:
                        language, value = pair.split(':')
                        result_dict[language] = float(value)

                    #print(f"index: {i}, score: {result_dict}")

                    if ('en' in result_dict.keys()) and (result_dict['en'] < 0.85):  ## cutoff
                        dropping_no_eng.append(i)

                    elif not 'en' in result_dict.keys():  ## cutoff
                        dropping_no_eng.append(i)

                    else:
                        pass



                else:
                    string = str(x).strip('[]')
                    language, value = string.split(':')
                    result_dict = {language: float(value)}
                    #print(f"index: {i}, score: {result_dict}.")

                    if not 'en' in result_dict.keys():
                        dropping_no_eng.append(i)

                    else:
                        pass

            except:
                print(f'Something was wrong here: {i}')
                dropping_no_eng.append(i)
        print(f"Before dropping length: {len(self.mil_and_genz)}")
        self.mil_and_genz.drop(dropping_no_eng, axis=0, inplace=True)
        self.mil_and_genz.reset_index(inplace=True)
        print(f"After dropping length: {len(self.mil_and_genz)}")

        print('Done language detecting.')

        return self.mil_and_genz

    def train_fasttext_model(self):
        """
        trains a fasttext model on the corpus and saves it
        """
        corpus = " ".join([" ".join(post) for post in self.mil_and_genz['post_tokenized']])
        with open("data/temp_corpus.txt", "w", encoding="utf-8") as file:
            file.write(corpus)
        self.model = fasttext.train_unsupervised("data/temp_corpus.txt")
        if not os.path.exists('result/'):
            os.makedirs('result/')
        self.model.save_model(self.output_model_path)
        return self.model

    def process_and_add_embedding(self):
        """
        get the average embedding for per post (document embedding)
        """
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
        """
        create a new column with document embedding averages
        """
        def calculate_average(lst):
            return sum(lst) / len(lst) if len(lst) > 0 else None
        self.mil_and_genz['doc_embedding_average'] = self.mil_and_genz['doc_embedding'].apply(calculate_average)

