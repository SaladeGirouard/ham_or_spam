import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

# Téléchargement des ressources NLTK
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Chargement des données
@st.cache_data
def load_data():
    dfspam = pd.read_csv('SMSSpamCollection.csv', sep="\t", header=None, names=['Type', 'comment'])
    return dfspam

dfspam = load_data()

# Définir les stopwords, la ponctuation et le lemmatiseur
stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# Fonction pour nettoyer le texte
def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in punctuation)
    tokens = nltk.word_tokenize(text)
    text = ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stopwords)
    return text

# Nettoyage des données
dfspam['comment_clean'] = dfspam['comment'].apply(clean_text)

# Préparation des données pour le modèle Scikit-learn
X = dfspam['comment_clean']
y = dfspam['Type']

# Fonction pour entraîner et évaluer un modèle
@st.cache_resource
def train_model(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    pipeline_model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', model)
    ])
    pipeline_model.fit(x_train, y_train)
    return pipeline_model

# Entraînement du modèle Naive Bayes
model_naive = MultinomialNB()
pipeline_model = train_model(model_naive, X, y)

# Préparation des données pour le modèle TensorFlow
dfspam['msg_type'] = dfspam['Type'].map({'ham': 0, 'spam': 1})
msg_label = dfspam['msg_type'].values
train_msg, test_msg, train_labels, test_labels = train_test_split(dfspam['comment_clean'], msg_label, test_size=0.2, random_state=22)

# Tokenisation et padding
max_len = 50
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
vocab_size = 500
tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
tokenizer.fit_on_texts(train_msg)

training_sequences = tokenizer.texts_to_sequences(train_msg)
training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(test_msg)
testing_padded = pad_sequences(testing_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

# Architecture du modèle TensorFlow
embedding_dim = 16
drop_value = 0.2
n_dense = 24

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(drop_value),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    training_padded, train_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, test_labels),
    callbacks=[early_stop],
    verbose=2
)

# Fonction pour prédire si un message est du spam ou non
def Spam_or_ham(msg: str, model_name: str, model):
    msg = clean_text(msg)
    msgarray = np.array([msg])

    if model_name == "TensorFlow":
        sequences = tokenizer.texts_to_sequences(msgarray)
        padded = pad_sequences(sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
        result = model.predict(padded, verbose=0)
        confidence = result[0][0] * 100
        if result[0][0] < 0.5:
            text = f'This message is a ham at {round(100 - confidence, 2)}% confidence'
        else:
            text = f'This message is a spam at {round(confidence, 2)}% confidence'
    else:
        prediction = pipeline_model.predict_proba(msgarray)
        confidence = prediction[0][0] * 100
        if prediction[0][0] > 0.5:
            text = f'This message is ham at {round(confidence, 2)}% confidence'
        else:
            text = f'This message is a spam at {round(100 - confidence, 2)}% confidence'
    return text

# Interface Streamlit
st.set_page_config(
    page_title="Ham or Spam?",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image("hamjam.png")
text_input = st.text_input('Entrez du texte :', key='input_text')
option = st.radio('On part sur quel algo ?', ('Scikit-Learn', 'TensorFlow', 'Ensemble'))

if st.button('Valider'):
    if text_input:
        if option == 'Scikit-Learn':
            st.write(Spam_or_ham(text_input, option, model_naive))
        elif option == 'TensorFlow':
            st.write(Spam_or_ham(text_input, option, model))
        elif option == 'Ensemble':
            msg = clean_text(text_input)
            msgarray = np.array([msg])
            sequences = tokenizer.texts_to_sequences(msgarray)
            padded = pad_sequences(sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
            result = model.predict(padded, verbose=0)
            result_skl = pipeline_model.predict_proba(msgarray)
            hamprob = (result_skl[0][0] + (1 - result[0][0])) / 2 * 100
            spamprob = (result_skl[0][1] + result[0][0]) / 2 * 100
            if hamprob > spamprob:
                st.write(f'This message is ham at {round(hamprob, 2)}% confidence')
            else:
                st.write(f'This message is spam at {round(spamprob, 2)}% confidence')
    else:
        st.write("Veuillez entrer du texte dans la zone de texte.")
