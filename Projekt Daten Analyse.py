nltk.download('stopwords')

import re 
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

import nltk
import ntlk.corpus 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


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
eStopWords.add('comcast') #Comcast als zusätzliches Stopwort, da dieses sehr häufig in den Beschwerden auftaucht aber keine relevante Bedeutung hat.

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



# LSA
lsa = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=15, random_state=42)
lsa_output = lsa.fit_transform(X)
# Neue Spalte für jede Komponente im DF
for i in range(lsa_output.shape[1]):
    df[f'LSA Topic {i}'] = lsa_output[:, i]


# LDA
lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.9, topic_word_prior=0.9)
lda_output = lda.fit_transform(X)
# Neue Spalte für jede Komponente im DF
for i in range(lda_output.shape[1]):
    df[f'LDA Topic {i}'] = lda_output[:, i]

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

# Coherence Score (Bestimmung der Anzahl von Themen)
coherence_model_lda = CoherenceModel(topics=topics, texts=df['Customer Complaint'], dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print('Coherence Score: ', coherence_lda)
print(topics)
#LDA Themen
for i, topic in enumerate(topics):
    print(f"Top words for topic {i}: {', '.join(topic)}")

