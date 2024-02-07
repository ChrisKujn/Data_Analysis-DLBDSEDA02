nltk.download('stopwords')

import re 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import ntlk.corpus 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Inhalt der CSV 'Comcast.csv' lesen / Data Frame (df) = pandas.read.csv
df = pd.read.csv ('Comcast.csv', engine = 'python')

# Nachdem der Inhalt der CSV ausgelesen wurde, wird der Inhalt der Spalte in Kleinbuchstaben umgewandelt.
# Relevant hierfür ist die Spalte 'Customer Complaint', da diese die Kundenbeschwerden enthält. 
df['Customer Complaint'] = df['Customer Complaint'].str.lower
# print df['Customer Complaint']

# Im nächsten Schritt werden die Wörter mithilfe der Funktion word-tokenize aus der Unterbibiliothek nltk.tokenize tokenisiert.
df['Customer Complaint'] = df['Customer Complaint'].apply(word_tokenize)

# eStopWords mit den englischen Stoppwörtern füllen, da der vorhandene Datensatz auf englisch ist 
eStopWords = set(stopwords.words('english'))
eStopWords.add('comcast')

# Bedeutungslose Wörter, auch Stoppwörter genannt, entfernen.
df['Customer Complaint'] = df['Customer Complaint'].apply(lambda x: [word for word in x if word not in eStopWords])
# print df['Customer Complaint']

# Die restlichen Wörter in die Grundform bringen (Stemming oder Lemmatisierung) 
lemmatizer = WordNetLemmatizer()
df['Customer Complaint'] = df['Customer Complaint'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Umwandlung in nummerische Vektoren
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Customer Complaint'].apply(' '.join))



# Mithilfe des Bag-of-Words-Ansatzes (BoW) wird der vorverarbeiteten Datensatz in nummerische Vektoren umgewandelt


# TF-IDF Ansatz. Beide Ansätze können in Python mithilfe des scikit-learn-Pakets umgesetzt dafür wird pandas als pd importiert.
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['Customer Complaint'].apply(' '.join))


# Coherence Score (Bestimmung der Anzahl von Themen)
# Word Embeddings oder zumindest N-Gram
# ggf. Custom Stoppwortfilter (sogenannte Domain-Filter)
