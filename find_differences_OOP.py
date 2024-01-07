from stylometry_functions import StylometryAnalyzer

# Load data and get vocabulary size
analyzer = StylometryAnalyzer(data_path='data/birth_year.csv')
df_birth_year = analyzer.df_birth_year
mil_and_genz, mil, genz, tokens_mil, tokens_genz, t = analyzer.process_data()
vocabulary_size_m = analyzer.vocabulary_size(tokens_mil)
vocabulary_size_gz = analyzer.vocabulary_size(tokens_genz)
vocabulary_size_total = analyzer.vocabulary_size(t)
print(vocabulary_size_m, vocabulary_size_gz, vocabulary_size_total)

# Create over and undersampling
genz_oversampled, tokens_genz_oversampled, mil_undersampled, tokens_mil_undersampled = analyzer.handle_imbalances(genz, mil)

# Compare token length distribution
generational_tokens = {'mil': [token.lower() for token_list in tokens_mil for token in token_list],
                       'genz': [token.lower() for token_list in tokens_genz for token in token_list]}
print('Token length distribution')
token_dict = analyzer.stylometry_comparison(generational_tokens, 'Token', mil, genz)
average_mil = sum(key * value for key, value in token_dict['mil'].items()) / sum(token_dict['mil'].values())
average_genz = sum(key * value for key, value in token_dict['genz'].items()) / sum(token_dict['genz'].values())
print('Average token length: ', average_mil, average_genz)


print('Token length distribution (genz oversampled)')
generational_tokens_os = {}
generational_tokens_os['mil'] = [token.lower() for token_list in tokens_mil for token in token_list]
generational_tokens_os['genz'] = [token.lower() for token_list in tokens_genz_oversampled for token in token_list]
token_dict_oversampled = analyzer.stylometry_comparison(generational_tokens_os, 'Token', mil, genz_oversampled)
average_mil_os = sum(key * value for key, value in token_dict_oversampled['mil'].items()) / sum(token_dict_oversampled['mil'].values())
average_genz_os = sum(key * value for key, value in token_dict_oversampled['genz'].items()) / sum(token_dict_oversampled['genz'].values())
print('average token length: ', average_mil_os, average_genz_os)

print('Token length distribution (mil undersampled)')
generational_tokens_us = {}
generational_tokens_us['mil'] = [token.lower() for token_list in tokens_mil_undersampled for token in token_list]
generational_tokens_us['genz'] = [token.lower() for token_list in tokens_genz for token in token_list]
token_dict_undersampled = analyzer.stylometry_comparison(generational_tokens_us, 'Token', mil_undersampled, genz)
average_mil_us = sum(key * value for key, value in token_dict_undersampled['mil'].items()) / sum(token_dict_undersampled['mil'].values())
average_genz_us = sum(key * value for key, value in token_dict_undersampled['genz'].items()) / sum(token_dict_undersampled['genz'].values())
print('average token length: ', average_mil_us, average_genz_us)

# compare word length distribution
print('Word length distribution')
generational_words = {}
generational_words['mil'] = [token.lower() for token_list in tokens_mil for token in token_list if token.isalpha()]
generational_words['genz'] = [token.lower() for token_list in tokens_genz for token in token_list if token.isalpha()]
word_dict = analyzer.stylometry_comparison(generational_words, 'Word', mil, genz)
agv_word_length_mil = sum(key * value for key, value in word_dict['mil'].items()) / sum(word_dict['mil'].values())
agv_word_length_genz = sum(key * value for key, value in word_dict['genz'].items()) / sum(word_dict['genz'].values())
print('average word length: ', agv_word_length_mil, agv_word_length_genz)

print('Word length distribution (genz oversampled)')
generational_words_os = {}
generational_words_os['mil'] = [token.lower() for token_list in tokens_mil for token in token_list if token.isalpha()]
generational_words_os['genz'] = [token.lower() for token_list in tokens_genz_oversampled for token in token_list if token.isalpha()]
word_dict_oversampled = analyzer.stylometry_comparison(generational_words_os, 'Word', mil, genz_oversampled)
agv_word_length_mil_os = sum(key * value for key, value in word_dict_oversampled['mil'].items()) / sum(word_dict_oversampled['mil'].values())
agv_word_length_genz_os = sum(key * value for key, value in word_dict_oversampled['genz'].items()) / sum(word_dict_oversampled['genz'].values())
print('average word length: ', agv_word_length_mil_os, agv_word_length_genz_os)

print('Word length distribution (mil undersampled)')
generational_words_us = {}
generational_words_us['mil'] = [token.lower() for token_list in tokens_mil_undersampled for token in token_list if token.isalpha()]
generational_words_us['genz'] = [token.lower() for token_list in tokens_genz for token in token_list if token.isalpha()]
word_dict_undersampled = analyzer.stylometry_comparison(generational_words_us, 'Word', mil_undersampled, genz)
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
docu_dict = analyzer.stylometry_comparison(generational_token_sentence, 'Document', mil, genz)
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
docu_dict_os = analyzer.stylometry_comparison(generational_token_sentence_os, 'Document', mil, genz_oversampled)
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
docu_dict_us = analyzer.stylometry_comparison(generational_token_sentence_us, 'Document', mil_undersampled, genz)
agv_docu_length_mil_us = sum(key * value for key, value in docu_dict_us['mil'].items()) / sum(docu_dict_us['mil'].values())
agv_docu_length_genz_us = sum(key * value for key, value in docu_dict_us['genz'].items()) / sum(docu_dict_us['genz'].values())
print('average document length: ', agv_docu_length_mil_us, agv_docu_length_genz_us)

# compare number of sentences distribution
print('Number of sentences distribution')
sent_dict = analyzer.stylometry_comparison(0,'Sentence', mil, genz)
avg_nr_sent_mil = sum(length * freq for length, freq in sent_dict['mil'].items()) / sum(sent_dict['mil'].values())
avg_nr_sent_genz = sum(length * freq for length, freq in sent_dict['genz'].items()) / sum(sent_dict['genz'].values())
print('average number of sentences: ', avg_nr_sent_mil, avg_nr_sent_genz)

print('Number of sentences distribution (genz oversampled)')
sent_dict_os = analyzer.stylometry_comparison(0,'Sentence', mil, genz_oversampled)
avg_nr_sent_mil_os = sum(length * freq for length, freq in sent_dict_os['mil'].items()) / sum(sent_dict_os['mil'].values())
avg_nr_sent_genz_os = sum(length * freq for length, freq in sent_dict_os['genz'].items()) / sum(sent_dict_os['genz'].values())
print('average number of sentences: ', avg_nr_sent_mil_os, avg_nr_sent_genz_os)

print('Number of sentences distribution (mil undersampled)')
sent_dict_us = analyzer.stylometry_comparison(0,'Sentence', mil_undersampled, genz)
avg_nr_sent_mil_us = sum(length * freq for length, freq in sent_dict_us['mil'].items()) / sum(sent_dict_us['mil'].values())
avg_nr_sent_genz_us = sum(length * freq for length, freq in sent_dict_us['genz'].items()) / sum(sent_dict_us['genz'].values())
print('average number of sentences: ', avg_nr_sent_mil_us, avg_nr_sent_genz_us)

# compare average sentence length per document distribution
print('Average sentence length distribution')
sent_len_dict = analyzer.sent_len_comparison('Sentence', mil, genz)
avg_sent_len_mil = sum(length * freq for length, freq in sent_len_dict['mil'].items()) / sum(sent_len_dict['mil'].values())
avg_sent_len_genz = sum(length * freq for length, freq in sent_len_dict['genz'].items()) / sum(sent_len_dict['genz'].values())
print('average average sentence length: ', avg_sent_len_mil, avg_sent_len_genz)

print('Average sentence length distribution (genz oversampled)')
sent_len_dict_os = analyzer.sent_len_comparison('Sentence', mil, genz_oversampled)
avg_sent_len_mil_os = sum(length * freq for length, freq in sent_len_dict_os['mil'].items()) / sum(sent_len_dict_os['mil'].values())
avg_sent_len_genz_os = sum(length * freq for length, freq in sent_len_dict_os['genz'].items()) / sum(sent_len_dict_os['genz'].values())
print('average average sentence length: ', avg_sent_len_mil_os, avg_sent_len_genz_os)

print('Average sentence length distribution (mil undersampled)')
sent_len_dict_us = analyzer.sent_len_comparison('Sentence', mil_undersampled, genz)
avg_sent_len_mil_us = sum(length * freq for length, freq in sent_len_dict_us['mil'].items()) / sum(sent_len_dict_us['mil'].values())
avg_sent_len_genz_us = sum(length * freq for length, freq in sent_len_dict_us['genz'].items()) / sum(sent_len_dict_us['genz'].values())
print('average average sentence length: ', avg_sent_len_mil_us, avg_sent_len_genz_us)

# compare stopwords distribution
print('Stopwords distribution')
stopwords_dict = analyzer.stopwords_comparison(generational_tokens)
print('Stopwords distribution (genz overersampled)')
stopwords_dict_os = analyzer.stopwords_comparison(generational_tokens_os)
print('Stopwords distribution (mil undersampled)')
stopwords_dict_us = analyzer.stopwords_comparison(generational_tokens_us)

# compare punctuation
print('Punctuation distribution')
punc_dict = analyzer.punctuation_comparison(generational_tokens)
print('Punctuation distribution (genz overersampled)')
punc_dict_os = analyzer.punctuation_comparison(generational_tokens_os)
print('Punctuation distribution (mil undersampled)')
punc_dict_us = analyzer.punctuation_comparison(generational_tokens_us)

# compare POS tags
print('POS distribution')
pos_dict = analyzer.parts_of_speech_comparison(generational_words)
print('POS distribution (genz overersampled)')
pos_dict_os = analyzer.parts_of_speech_comparison(generational_words_os)
print('POS distribution (mil undersampled)')
pos_dict_us = analyzer.parts_of_speech_comparison(generational_words_us)
