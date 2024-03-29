# Language and AI - Generational Cohort Profiling on Gen Z and Millennial Reddit Posts

Repository for the experiments described in "Generational Cohort Profiling on Gen Z and Millennial Reddit Posts".

## Overview

- [Paper Details](#PaperDetails)
  - [tl;dr](#tl;dr)
  - [Reproduction](#Reproduction)
  - [Dependencies](#Dependencies)
- [Extensions](#Extensions)
  - [Adding Data](#AddingData)
  - [Changing Classifiers](#ChangingClassifiers)

## PaperDetails

### tl;dr

We :
- Classified Reddit posts into categories Millennial and Genz, based on stylometric features. 
- Preprocessed the posts by filtering on language. 
- Compared stylometric features of Millennials to those of Genzers. 
- Used important stylometric features to train a logistic regression, naive bayes and SVM model on Reddit posts. 

### Reproduction

To reproduce the results in "Generational Cohort Profiling on Gen Z and Millennial Reddit Posts", run 'experiment.py'. It is assumed that there is a \data folder containing the provided auhtor_birthyear table and is stored as 'birth_year.csv'. Running 'find_differences_OOP.py' provides additional plots and statistics that back up decisions made to determine stylometric features that were different from Millennials and Genzers. 
> The code was tested with Python 3.8.8 on Windows.

'experiment.py' imports the classes Vectorizer, FasttextEmbedding and ModelEvaluator from 'vectorization_functions.py', 'embeddings_functions.py' and 'classification_functions.py' respectively. 
The methods in Vectorizer distills, after dropping posts with <85% English, per Reddit post the document length, number of sentences, average sentence length, POS tags dictionary and the count of occurrences of the following tokens: 'a', 'and', 'you', 'is', '?', '"', '/', '#', '!'. 
The methods in FasttextEmbedding create, after dropping posts with <85% English, word embeddings using Fasttext, average those to get document embeddings and take the average of the document embeddings as an additional feature.
The methods in ModelEvaluator apply feature selection, train a Logistic Regression, Naive Bayes and SVM model and compare it to the Logistic Regression majority baseline model and the default Logistic Regression model. 

### Dependencies

The code was tested using these libraries and versions:

```
pandas          1.2.4
fasttext        0.9.2
numpy           1.20.1
seaborn         0.11.1
matplotlib      3.3.4
scikit-learn    0.24.1
nltk            3.6.1
langdetect      1.0.9
```

## Extensions

### AddingData
This code can be used in different data, as long as there is a file named 'birth_year.csv' stored in a \data folder. This file should contain information in the columns auhtor_ID, post, and birth_year. The information stored in auhtor_ID can be of any form, as long as it is representative of the author of the post. Post should be a string containing the post written by the author. birth_year should contain the birthyear of the author as integer and should have at least one author with birthyear in the range (1986,1996] and one author with birthyear in range (1996, 2006]. 

### ChangingClassifiers
The classification years can be adapted by changing the values in the load_data methods from the Vectorizer and ModelEvaluator classes, which can be found in 'vectorization_functions.py' and 'classification_functions.py' respectively. 