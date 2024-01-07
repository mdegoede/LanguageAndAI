import pandas as pd
import nltk
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.utils import resample
from nltk.tokenize import sent_tokenize, word_tokenize

def process_data(data):
    # structure the data, adding millenials and genz classes and tokenize the posts for mil and genz separately
    df_birth_year = pd.read_csv(data)
    mil_and_genz = df_birth_year[(1986 < df_birth_year['birth_year']) & (df_birth_year['birth_year'] <= 2006)]
    mil_and_genz['binary_birth_year'] = mil_and_genz['birth_year']
    mil_and_genz.loc[(1986 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 1096), 'binary_birth_year'] = 1
    mil_and_genz.loc[(1096 < mil_and_genz['birth_year']) & (mil_and_genz['birth_year'] <= 2006), 'binary_birth_year'] = 0
    mil_and_genz = mil_and_genz.reset_index(drop=True)

    mil = df_birth_year[(1986 < df_birth_year['birth_year']) & (df_birth_year['birth_year'] <= 1996)]
    genz = df_birth_year[(1996 < df_birth_year['birth_year']) & (df_birth_year['birth_year'] <= 2006)]
    tokens_mil = mil.post.str.findall('\w+|[^\w\s]')
    tokens_genz = genz.post.str.findall('\w+|[^\w\s]')
    t = df_birth_year.post.str.findall('\w+|[^\w\s]')

    return df_birth_year, mil_and_genz, mil, genz, tokens_mil, tokens_genz, t

def vocabulary_size(df):
    # vocabulary size
    flat_tokens = [token.lower() for sublist in df for token in sublist]
    vocabulary = set(flat_tokens)
    return len(vocabulary)


def handle_imbalances(genz_df, mil_df):
    # over and undersampling
    genz_oversampled = resample(genz_df, replace=True, n_samples=len(mil_df), random_state=42)
    tokens_genz_oversampled = genz_oversampled.post.str.findall('\w+|[^\w\s]')

    mil_undersampled = resample(mil_df, replace=False, n_samples=len(genz_df), random_state=42)
    tokens_mil_undersampled = mil_undersampled.post.str.findall('\w+|[^\w\s]')

    return genz_oversampled, tokens_genz_oversampled, mil_undersampled, tokens_mil_undersampled

def stylometry_comparison(corpus=0, task='Word', df_mil=0, df_genz=0):
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

def sent_len_comparison(task='Word', df_mil=0, df_genz=0):
    len_by_author_dict = {}
    gen1 = 'mil'
    gen2 = 'genz'
    lengths_mil = df_mil['post'].apply(lambda text: round(sum(1 for sentence in sent_tokenize(text) for word in word_tokenize(sentence) if word.isalpha()) / max(1, len(sent_tokenize(text))),0))
    lengths_genz = df_genz['post'].apply(lambda text: round(sum(1 for sentence in sent_tokenize(text) for word in word_tokenize(sentence) if word.isalpha()) / max(1, len(sent_tokenize(text))),0))
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

def stopwords_comparison(corpus):
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

def punctuation_comparison(corpus):
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

def parts_of_speech_comparison(corpus):
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

# load data and get vocabulary size
df_birth_year, mil_and_genz, mil, genz, tokens_mil, tokens_genz, t = process_data('data/birth_year.csv')
vocabulary_size_m = vocabulary_size(tokens_mil)
vocabulary_size_gz = vocabulary_size(tokens_genz)
vocabulary_size = vocabulary_size(t)
print(vocabulary_size_m, vocabulary_size_gz, vocabulary_size)

# create over and undersampling
genz_oversampled, tokens_genz_oversampled, mil_undersampled, tokens_mil_undersampled = handle_imbalances(genz, mil)

# compare token length distribution
generational_tokens = {}
generational_tokens['mil'] = [token.lower() for token_list in tokens_mil for token in token_list]
generational_tokens['genz'] = [token.lower() for token_list in tokens_genz for token in token_list]
print('Token length distribution')
token_dict = stylometry_comparison(generational_tokens, 'Token')
average_mil = sum(key * value for key, value in token_dict['mil'].items()) / sum(token_dict['mil'].values())
average_genz = sum(key * value for key, value in token_dict['genz'].items()) / sum(token_dict['genz'].values())
print('average token length: ', average_mil, average_genz)

print('Token length distribution (genz oversampled)')
generational_tokens_os = {}
generational_tokens_os['mil'] = [token.lower() for token_list in tokens_mil for token in token_list]
generational_tokens_os['genz'] = [token.lower() for token_list in tokens_genz_oversampled for token in token_list]
token_dict_oversampled = stylometry_comparison(generational_tokens_os, 'Token')
average_mil_os = sum(key * value for key, value in token_dict_oversampled['mil'].items()) / sum(token_dict_oversampled['mil'].values())
average_genz_os = sum(key * value for key, value in token_dict_oversampled['genz'].items()) / sum(token_dict_oversampled['genz'].values())
print('average token length: ', average_mil_os, average_genz_os)

print('Token length distribution (mil undersampled)')
generational_tokens_us = {}
generational_tokens_us['mil'] = [token.lower() for token_list in tokens_mil_undersampled for token in token_list]
generational_tokens_us['genz'] = [token.lower() for token_list in tokens_genz for token in token_list]
token_dict_undersampled = stylometry_comparison(generational_tokens_us, 'Token')
average_mil_us = sum(key * value for key, value in token_dict_undersampled['mil'].items()) / sum(token_dict_undersampled['mil'].values())
average_genz_us = sum(key * value for key, value in token_dict_undersampled['genz'].items()) / sum(token_dict_undersampled['genz'].values())
print('average token length: ', average_mil_us, average_genz_us)

# compare word length distribution
print('Word length distribution')
generational_words = {}
generational_words['mil'] = [token.lower() for token_list in tokens_mil for token in token_list if token.isalpha()]
generational_words['genz'] = [token.lower() for token_list in tokens_genz for token in token_list if token.isalpha()]
word_dict = stylometry_comparison(generational_words, 'Word')
agv_word_length_mil = sum(key * value for key, value in word_dict['mil'].items()) / sum(word_dict['mil'].values())
agv_word_length_genz = sum(key * value for key, value in word_dict['genz'].items()) / sum(word_dict['genz'].values())
print('average word length: ', agv_word_length_mil, agv_word_length_genz)

print('Word length distribution (genz oversampled)')
generational_words_os = {}
generational_words_os['mil'] = [token.lower() for token_list in tokens_mil for token in token_list if token.isalpha()]
generational_words_os['genz'] = [token.lower() for token_list in tokens_genz_oversampled for token in token_list if token.isalpha()]
word_dict_oversampled = stylometry_comparison(generational_words_os, 'Word')
agv_word_length_mil_os = sum(key * value for key, value in word_dict_oversampled['mil'].items()) / sum(word_dict_oversampled['mil'].values())
agv_word_length_genz_os = sum(key * value for key, value in word_dict_oversampled['genz'].items()) / sum(word_dict_oversampled['genz'].values())
print('average word length: ', agv_word_length_mil_os, agv_word_length_genz_os)

print('Word length distribution (mil undersampled)')
generational_words_us = {}
generational_words_us['mil'] = [token.lower() for token_list in tokens_mil_undersampled for token in token_list if token.isalpha()]
generational_words_us['genz'] = [token.lower() for token_list in tokens_genz for token in token_list if token.isalpha()]
word_dict_undersampled = stylometry_comparison(generational_words_us, 'Word')
agv_word_length_mil_us = sum(key * value for key, value in word_dict_undersampled['mil'].items()) / sum(word_dict_undersampled['mil'].values())
agv_word_length_genz_us = sum(key * value for key, value in word_dict_undersampled['genz'].items()) / sum(word_dict_undersampled['genz'].values())
print('average word length: ', agv_word_length_mil_us, agv_word_length_genz_us)

# compare document length distribution
print('Document length distribution')
generational_token_sentence = {}
# get all tokens of the whole reddit in one list
generational_token_sentence['mil'] = []
for token_list in tokens_mil:
    word_sentence = []
    for token in token_list:
        if token.isalpha():
            word_sentence.append(token)
    generational_token_sentence['mil'].append(word_sentence)
generational_token_sentence['genz'] = []
for token_list in tokens_genz:
    word_sentence = []
    for token in token_list:
        if token.isalpha():
            word_sentence.append(token)
    generational_token_sentence['genz'].append(word_sentence)
docu_dict = stylometry_comparison(generational_token_sentence, 'Document')
agv_docu_length_mil = sum(key * value for key, value in docu_dict['mil'].items()) / sum(docu_dict['mil'].values())
agv_docu_length_genz = sum(key * value for key, value in docu_dict['genz'].items()) / sum(docu_dict['genz'].values())
print('average document length: ', agv_docu_length_mil, agv_docu_length_genz)

print('Document length distribution (genz oversampled)')
generational_token_sentence_os = {}
generational_token_sentence_os['mil'] = []
for token_list in tokens_mil:
    word_sentence = []
    for token in token_list:
        if token.isalpha():
            word_sentence.append(token)
    generational_token_sentence_os['mil'].append(word_sentence)
generational_token_sentence_os['genz'] = []
for token_list in tokens_genz_oversampled:
    word_sentence = []
    for token in token_list:
        if token.isalpha():
            word_sentence.append(token)
    generational_token_sentence_os['genz'].append(word_sentence)
docu_dict_os = stylometry_comparison(generational_token_sentence_os, 'Document')
agv_docu_length_mil_os = sum(key * value for key, value in docu_dict_os['mil'].items()) / sum(docu_dict_os['mil'].values())
agv_docu_length_genz_os = sum(key * value for key, value in docu_dict_os['genz'].items()) / sum(docu_dict_os['genz'].values())
print('average document length: ', agv_docu_length_mil_os, agv_docu_length_genz_os)

print('Document length distribution (mil undersampled)')
generational_token_sentence_us = {}
generational_token_sentence_us['mil'] = []
for token_list in tokens_mil_undersampled:
    word_sentence = []
    for token in token_list:
        if token.isalpha():
            word_sentence.append(token)
    generational_token_sentence_us['mil'].append(word_sentence)
generational_token_sentence_us['genz'] = []
for token_list in tokens_genz:
    word_sentence = []
    for token in token_list:
        if token.isalpha():
            word_sentence.append(token)
    generational_token_sentence_us['genz'].append(word_sentence)
docu_dict_us = stylometry_comparison(generational_token_sentence_us, 'Document')
agv_docu_length_mil_us = sum(key * value for key, value in docu_dict_us['mil'].items()) / sum(docu_dict_us['mil'].values())
agv_docu_length_genz_us = sum(key * value for key, value in docu_dict_us['genz'].items()) / sum(docu_dict_us['genz'].values())
print('average document length: ', agv_docu_length_mil_us, agv_docu_length_genz_us)

# compare number of sentences distribution
print('Number of sentences distribution')
sent_dict = stylometry_comparison(0,'Sentence', mil, genz)
avg_nr_sent_mil = sum(length * freq for length, freq in sent_dict['mil'].items()) / sum(sent_dict['mil'].values())
avg_nr_sent_genz = sum(length * freq for length, freq in sent_dict['genz'].items()) / sum(sent_dict['genz'].values())
print('average number of sentences: ', avg_nr_sent_mil, avg_nr_sent_genz)

print('Number of sentences distribution (genz oversampled)')
sent_dict_os = stylometry_comparison(0,'Sentence', mil, genz_oversampled)
avg_nr_sent_mil_os = sum(length * freq for length, freq in sent_dict_os['mil'].items()) / sum(sent_dict_os['mil'].values())
avg_nr_sent_genz_os = sum(length * freq for length, freq in sent_dict_os['genz'].items()) / sum(sent_dict_os['genz'].values())
print('average number of sentences: ', avg_nr_sent_mil_os, avg_nr_sent_genz_os)

print('Number of sentences distribution (mil undersampled)')
sent_dict_us = stylometry_comparison(0,'Sentence', mil_undersampled, genz)
avg_nr_sent_mil_us = sum(length * freq for length, freq in sent_dict_us['mil'].items()) / sum(sent_dict_us['mil'].values())
avg_nr_sent_genz_us = sum(length * freq for length, freq in sent_dict_us['genz'].items()) / sum(sent_dict_us['genz'].values())
print('average number of sentences: ', avg_nr_sent_mil_us, avg_nr_sent_genz_us)

# compare average sentence length per document distribution
print('Average sentence length distribution')
sent_len_dict = sent_len_comparison('Sentence', mil, genz)
avg_sent_len_mil = sum(length * freq for length, freq in sent_len_dict['mil'].items()) / sum(sent_len_dict['mil'].values())
avg_sent_len_genz = sum(length * freq for length, freq in sent_len_dict['genz'].items()) / sum(sent_len_dict['genz'].values())
print('average average sentence length: ', avg_sent_len_mil, avg_sent_len_genz)

print('Average sentence length distribution (genz oversampled)')
sent_len_dict_os = sent_len_comparison('Sentence', mil, genz_oversampled)
avg_sent_len_mil_os = sum(length * freq for length, freq in sent_len_dict_os['mil'].items()) / sum(sent_len_dict_os['mil'].values())
avg_sent_len_genz_os = sum(length * freq for length, freq in sent_len_dict_os['genz'].items()) / sum(sent_len_dict_os['genz'].values())
print('average average sentence length: ', avg_sent_len_mil_os, avg_sent_len_genz_os)

print('Average sentence length distribution (mil undersampled)')
sent_len_dict_us = sent_len_comparison('Sentence', mil_undersampled, genz)
avg_sent_len_mil_us = sum(length * freq for length, freq in sent_len_dict_us['mil'].items()) / sum(sent_len_dict_us['mil'].values())
avg_sent_len_genz_us = sum(length * freq for length, freq in sent_len_dict_us['genz'].items()) / sum(sent_len_dict_us['genz'].values())
print('average average sentence length: ', avg_sent_len_mil_us, avg_sent_len_genz_us)

# compare stopwords distribution
print('Stopwords distribution')
stopwords_dict = stopwords_comparison(generational_tokens)
print('Stopwords distribution (genz overersampled)')
stopwords_dict_os = stopwords_comparison(generational_tokens_os)
print('Stopwords distribution (mil undersampled)')
stopwords_dict_us = stopwords_comparison(generational_tokens_us)

# compare punctuation
print('Punctuation distribution')
punc_dict = punctuation_comparison(generational_tokens)
print('Punctuation distribution (genz overersampled)')
punc_dict_os = punctuation_comparison(generational_tokens_os)
print('Punctuation distribution (mil undersampled)')
punc_dict_us = punctuation_comparison(generational_tokens_us)

# compare POS tags
print('POS distribution')
pos_dict = parts_of_speech_comparison(generational_words)
print('POS distribution (genz overersampled)')
pos_dict_os = parts_of_speech_comparison(generational_words_os)
print('POS distribution (mil undersampled)')
pos_dict_us = parts_of_speech_comparison(generational_words_us)
