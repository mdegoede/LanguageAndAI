import pandas as pd
import fasttext
import numpy as np

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

def train_fasttext_model(mil_and_genz, output_model_path):
    # trains a fasttext model on the corpus and saves it
    corpus = " ".join([" ".join(post) for post in mil_and_genz['post_tokenized']])
    with open("data/temp_corpus.txt", "w", encoding="utf-8") as file:
        file.write(corpus)
    model = fasttext.train_unsupervised("data/temp_corpus.txt")
    model.save_model(output_model_path)

def process_and_add_embedding(dataframe, model):
    # get the average embedding for per post (document embedding)
    def get_average_embedding(tokens):
        embeddings = [model.get_word_vector(word) for word in tokens if word in model.words]
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            avg_embedding = np.zeros(model.get_dimension())
        return avg_embedding

    dataframe['doc_embedding'] = dataframe['post_tokenized'].apply(get_average_embedding)
    return dataframe

def calculate_and_add_average_column(dataframe):
    # create a new column with document embedding averages
    def calculate_average(lst):
        return sum(lst) / len(lst) if len(lst) > 0 else None
    dataframe['doc_embedding_average'] = dataframe['doc_embedding'].apply(calculate_average)
    return dataframe

mil_and_genz = load_data('data/birth_year.csv')
train_fasttext_model(mil_and_genz, 'result/trained_fasttext_embeddings.bin') # skip this if you run the code for a second time
model = fasttext.load_model("result/trained_fasttext_embeddings.bin")
mil_and_genz = process_and_add_embedding(mil_and_genz, model)
mil_and_genz = calculate_and_add_average_column(mil_and_genz)
mil_and_genz.to_csv('data/mil_and_genz.csv', index=False)









