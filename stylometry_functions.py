import pandas as pd
import nltk
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.utils import resample
from nltk.tokenize import sent_tokenize, word_tokenize

class StylometryAnalyzer:
    def __init__(self, data_path):
        self.df_birth_year = pd.read_csv(data_path)
        self.mil_and_genz = self.process_data()

    def process_data(self):
        mil_and_genz = self.df_birth_year[(1986 < self.df_birth_year['birth_year']) & (self.df_birth_year['birth_year'] <= 2006)]
        mil_and_genz['binary_birth_year'] = mil_and_genz['birth_year']
        mil_and_genz.loc[(1986 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 1096), 'binary_birth_year'] = 1
        mil_and_genz.loc[(1096 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 2006), 'binary_birth_year'] = 0
        mil_and_genz = mil_and_genz.reset_index(drop=True)

        mil = self.df_birth_year[(1986 < self.df_birth_year['birth_year']) & (self.df_birth_year['birth_year'] <= 1996)]
        genz = self.df_birth_year[(1996 < self.df_birth_year['birth_year']) & (self.df_birth_year['birth_year'] <= 2006)]
        tokens_mil = mil.post.str.findall('\w+|[^\w\s]')
        tokens_genz = genz.post.str.findall('\w+|[^\w\s]')
        t = self.df_birth_year.post.str.findall('\w+|[^\w\s]')

        return mil_and_genz, mil, genz, tokens_mil, tokens_genz, t

    def vocabulary_size(self, df):
        flat_tokens = [token.lower() for sublist in df for token in sublist]
        vocabulary = set(flat_tokens)
        return len(vocabulary)

    def handle_imbalances(self, genz_df, mil_df):
        genz_oversampled = resample(genz_df, replace=True, n_samples=len(mil_df), random_state=42)
        tokens_genz_oversampled = genz_oversampled.post.str.findall('\w+|[^\w\s]')

        mil_undersampled = resample(mil_df, replace=False, n_samples=len(genz_df), random_state=42)
        tokens_mil_undersampled = mil_undersampled.post.str.findall('\w+|[^\w\s]')

        return genz_oversampled, tokens_genz_oversampled, mil_undersampled, tokens_mil_undersampled

    def stylometry_comparison(self, corpus, task, df_mil, df_genz):
        len_by_author_dict = {}
        if task == 'Sentence':
            gen1 = 'mil'
            gen2 = 'genz'
            lengths_mil = [len(nltk.sent_tokenize(text)) for text in df_mil['post']]
            lengths_genz = [len(nltk.sent_tokenize(text)) for text in df_genz['post']]
        else:
            generations = list(corpus.keys())
            gen1, gen2 = generations
            lengths_mil = [len(token) for token in corpus[gen1]]
            lengths_genz = [len(token) for token in corpus[gen2]]
        len_by_author_dict[gen1] = nltk.FreqDist(lengths_mil)
        len_by_author_dict[gen2] = nltk.FreqDist(lengths_genz)

        # extract the 15 highest token lengths for both generations
        highest_frequency = list((len_by_author_dict[gen1] + len_by_author_dict[gen2]).keys())[:15]
        df = pd.DataFrame({
            'Frequency': highest_frequency,
            gen1: [len_by_author_dict[gen1][token] for token in highest_frequency],
            gen2: [len_by_author_dict[gen2][token] for token in highest_frequency]
        })
        # sort the DataFrame by the frequencies
        df = df.sort_values(by=['Frequency'], ascending=True)

        plt.figure(figsize=(10, 6))
        plt.plot(df['Frequency'], df[gen1], label=gen1)
        plt.plot(df['Frequency'], df[gen2], label=gen2)
        plt.xlabel(f'{task} length')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of 15 highest frequencies of {task} length')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()
        return len_by_author_dict

    def sent_len_comparison(self, task, df_mil, df_genz):
        len_by_author_dict = {}
        gen1 = 'mil'
        gen2 = 'genz'
        lengths_mil = df_mil['post'].apply(lambda text: round(
            sum(1 for sentence in sent_tokenize(text) for word in word_tokenize(sentence) if word.isalpha()) / max(1,
                                                                                                                   len(sent_tokenize(
                                                                                                                       text))),
            0))
        lengths_genz = df_genz['post'].apply(lambda text: round(
            sum(1 for sentence in sent_tokenize(text) for word in word_tokenize(sentence) if word.isalpha()) / max(1,
                                                                                                                   len(sent_tokenize(
                                                                                                                       text))),
            0))
        len_by_author_dict[gen1] = nltk.FreqDist(lengths_mil)
        len_by_author_dict[gen2] = nltk.FreqDist(lengths_genz)

        # extract the 15 highest token lengths for both generations
        highest_frequency = list((len_by_author_dict[gen1] + len_by_author_dict[gen2]).keys())[:15]
        df = pd.DataFrame({
            'Frequency': highest_frequency,
            gen1: [len_by_author_dict[gen1][token] for token in highest_frequency],
            gen2: [len_by_author_dict[gen2][token] for token in highest_frequency]
        })
        # sort the DataFrame by the frequencies
        df = df.sort_values(by=['Frequency'], ascending=True)

        plt.figure(figsize=(10, 6))
        plt.plot(df['Frequency'], df[gen1], label=gen1)
        plt.plot(df['Frequency'], df[gen2], label=gen2)
        plt.xlabel(f'{task} length')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of 15 highest frequencies of average {task} length')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()
        return len_by_author_dict

    def stopwords_comparison(self, corpus):
        stopwords_by_author_dict = {}
        stop_words = set(stopwords.words('english'))
        generations = list(corpus.keys())
        gen1, gen2 = generations
        stopwords_gen1 = [word for word in corpus[gen1] if word in stop_words]
        stopwords_gen2 = [word for word in corpus[gen2] if word in stop_words]
        stopwords_by_author_dict[gen1] = nltk.FreqDist(stopwords_gen1)
        stopwords_by_author_dict[gen2] = nltk.FreqDist(stopwords_gen2)

        # Extract the 50 most common stopwords for both generations
        common_stopwords = list((stopwords_by_author_dict[gen1] + stopwords_by_author_dict[gen2]).keys())[:50]
        df = pd.DataFrame({
            'Stopword': common_stopwords,
            gen1: [stopwords_by_author_dict[gen1][word] for word in common_stopwords],
            gen2: [stopwords_by_author_dict[gen2][word] for word in common_stopwords]
        })
        # sort the DataFrame by the sum of frequencies in both generations
        df = df.sort_values(by=[gen1, gen2], ascending=False)

        plt.figure(figsize=(10, 6))
        plt.plot(df['Stopword'], df[gen1], label=gen1)
        plt.plot(df['Stopword'], df[gen2], label=gen2)
        plt.xlabel('Stopwords')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of 50 Most Common Stopwords - {gen1} vs. {gen2}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()
        return stopwords_by_author_dict

    def punctuation_comparison(self, corpus):
        punctuation_by_author_dict = {}
        generations = list(corpus.keys())
        gen1, gen2 = generations
        punctuation_gen1 = [word for word in corpus[gen1] if word in string.punctuation]
        punctuation_gen2 = [word for word in corpus[gen2] if word in string.punctuation]
        punctuation_by_author_dict[gen1] = nltk.FreqDist(punctuation_gen1)
        punctuation_by_author_dict[gen2] = nltk.FreqDist(punctuation_gen2)

        # Extract the 50 most common punctuation for both generations
        common_stopwords = list((punctuation_by_author_dict[gen1] + punctuation_by_author_dict[gen2]).keys())[:50]
        df = pd.DataFrame({
            'Stopword': common_stopwords,
            gen1: [punctuation_by_author_dict[gen1][word] for word in common_stopwords],
            gen2: [punctuation_by_author_dict[gen2][word] for word in common_stopwords]
        })
        # sort the DataFrame by the sum of frequencies in both generations
        df = df.sort_values(by=[gen1, gen2], ascending=False)

        plt.figure(figsize=(10, 6))
        plt.plot(df['Stopword'], df[gen1], label=gen1)
        plt.plot(df['Stopword'], df[gen2], label=gen2)
        plt.xlabel('Punctuation symbols')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of 50 Most Common Punctuation symbols - {gen1} vs. {gen2}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()
        return punctuation_by_author_dict

    def parts_of_speech_comparison(self, corpus):
        pos_by_author_dict = {}
        generations = list(corpus.keys())
        gen1, gen2 = generations
        punctuation_gen1 = [pos[1] for pos in nltk.pos_tag(corpus[gen1])]
        punctuation_gen2 = [pos[1] for pos in nltk.pos_tag(corpus[gen2])]
        pos_by_author_dict[gen1] = nltk.FreqDist(punctuation_gen1)
        pos_by_author_dict[gen2] = nltk.FreqDist(punctuation_gen2)

        # Extract the POSses for both generations
        common_stopwords = list((pos_by_author_dict[gen1] + pos_by_author_dict[gen2]).keys())
        df = pd.DataFrame({
            'Stopword': common_stopwords,
            gen1: [pos_by_author_dict[gen1][word] for word in common_stopwords],
            gen2: [pos_by_author_dict[gen2][word] for word in common_stopwords]
        })
        # sort the DataFrame by the sum of frequencies in both generations
        df = df.sort_values(by=[gen1, gen2], ascending=False)

        plt.figure(figsize=(10, 6))
        plt.plot(df['Stopword'], df[gen1], label=gen1)
        plt.plot(df['Stopword'], df[gen2], label=gen2)
        plt.xlabel('POS tags')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of POS tags - {gen1} vs. {gen2}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()
        return pos_by_author_dict