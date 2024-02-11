import re 
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


# Inhalt der CSV 'Comcast.csv' lesen
df = pd.read_csv ('Comcast.csv')
# Inhalt der Spalte 'Customer Complaint in Kleinbuchstaben umwandeln 
df['Customer Complaint'] = df['Customer Complaint'].str.lower()
# print (df['Customer Complaint'].head(5))


# Wörter mithilfe der Funktion word-tokenize aus der Unterbibiliothek nltk.tokenize tokenisiert.
df['Customer Complaint'] = df['Customer Complaint'].apply(word_tokenize)
# print (df['Customer Complaint'].head(5))


# eStopWords mit den englischen Stoppwörtern füllen
eStopWords = set(stopwords.words('english'))
#Comcast als zusätzliches Stopwort
eStopWords.add('comcast')
# Stoppwörter entfernen
df['Customer Complaint'] = df['Customer Complaint'].apply(lambda x: [word for word in x if word not in eStopWords])
# print (eStopWords)


# Wörter in die Grundform bringen (Stemming, Lemmatisierung) 
lemmatizer = WordNetLemmatizer()
df['Customer Complaint'] = df['Customer Complaint'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
# print (df['Customer Complaint'].head(5))


# Bag-of-Words-Vektorisierers
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(df['Customer Complaint'].apply(' '.join))
# print(X_bow[:5])


# TF-IDF Ansatz
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['Customer Complaint'].apply(' '.join))
# print(X_tfidf[:5])


# LSA
lsa = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=15, random_state=42)
lsa_output = lsa.fit_transform(X_tfidf)
# Neue Spalte für jede Komponente im Data frame
for i in range(lsa_output.shape[1]):
    df[f'LSA Topic {i}'] = lsa_output[:, i]
# print (lsa_output)
    

# LDA
lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.9, topic_word_prior=0.9)
lda_output = lda.fit_transform(X_tfidf)
# Neue Spalte für jede Komponente im Data frame
for i in range(lda_output.shape[1]):
    df[f'LDA Topic {i}'] = lda_output[:, i]
# print (lda_output)


# Verzeichnis für die Themen erstellen
dictionary = Dictionary(df['Customer Complaint'])


# Umwandlung in eine vektorisierte Form durch Berechnung des "Frequency counts"
corpus = [dictionary.doc2bow(doc) for doc in df['Customer Complaint']]


# Themen extrahieren
n_top_words = 3 
topics = []
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    topics.append(top_features)
# print (topics)


# Coherence Score
coherence_model_lda = CoherenceModel(topics=topics, texts=df['Customer Complaint'], dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()


#LDA Themen ausgeben
for i, topic in enumerate(topics):
    print(f"Top words for topic {i}: {', '.join(topic)}")


# Berechnung des Coherence Scores für LSA mithilfe von c_v measure
topics = [[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] for topic in lsa.components_]
coherence_model_lsa = CoherenceModel(topics=topics, texts=df['Customer Complaint'], dictionary=dictionary, coherence='c_v')
coherence_lsa = coherence_model_lsa.get_coherence()
print('Coherence Score: ', coherence_lsa)
# print(topics)


#LSA Themen ausgeben
for i, topic in enumerate(topics):
    print(f"Top words for topic {i}: {', '.join(topic)}")


# Datei für die Ausgabe der Ergebnisse erzeugen
df.to_csv('Comcast_Ergebnisse.csv')
