import re 
import pandas as pd

import ntlk.corpus 
from nltk.tokenize import word_tokenize


# Inhalt der CSV 'Comcast.csv' lesen / Data Frame (df) = pandas.read.csv
df = pd.read.csv ('Comcast.csv', engine = 'python')

# Nachdem der Inhalt der CSV ausgelesen wurde, wird der Inhalt der Spalte in Kleinbuchstaben umgewandelt.
# Relevant hierfür ist die Spalte 'Customer Complaint', da diese die Kundenbeschwerden enthält. 
df['Customer Complaint'] = df['Customer Complaint'].str.lower

# Im nächsten Schritt werden die Wörter mithilfe der Funktion word-tokenize aus der Unterbibiliothek nltk.tokenize tokenisiert.
df['Customer Complaint'] = df['Customer Complaint'].apply(word_tokenize)
