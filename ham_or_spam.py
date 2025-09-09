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
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

# Téléchargement des ressources NLTK (une seule fois)
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# 1. Chargement du CSV (mis en cache)
@st.cache_data
def load_data():
    dfspam = pd.read_csv('SMSSpamCollection.csv', sep="\t", header=None, names=['Type', 'comment'])
    dfspam = dfspam.dropna(subset=['Type', 'comment'])
    return dfspam

dfspam = load_data()

# 2. Nettoyage des données et préparation des features
stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in punctuation)
    tokens = nltk.word_tokenize(text)
    text = ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stopwords)
    return text

dfspam['comment_clean'] = dfspam['comment'].apply(clean_text)

# 3. Préparation des données pour les modèles
X = dfspam['comment_clean'].values
y = dfspam['Type'].replace({'ham': 0, 'spam': 1}).values  # Conversion en 0/1 pour éviter les problèmes de stratification

# 4. Entraînement des modèles (mis en cache)
@st.cache_resource
def train_models(X, y):
    # Modèle Scikit-Learn (Naive Bayes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline_nb = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    pipeline_nb.fit(X_train, y_train)

    # Modèle TensorFlow
    tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences, maxlen=50, padding="post", truncating="post")

    model_tf = Sequential([
        Embedding(500, 16, input_length=50),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model_tf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_tf.fit(
        train_padded, y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
        verbose=0
    )

    return pipeline_nb, model_tf, tokenizer

# 5. Chargement/entraînement des modèles (une seule fois)
pipeline_nb, model_tf, tokenizer = train_models(X, y)

# 6. Fonction de prédiction
def predict_message(msg, model_type):
    msg_clean = clean_text(msg)
    msg_array = np.array([msg_clean])

    if model_type == "TensorFlow":
        sequence = tokenizer.texts_to_sequences(msg_array)
        padded = pad_sequences(sequence, maxlen=50, padding="post", truncating="post")
        prob = model_tf.predict(padded, verbose=0)[0][0]
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
        label = "spam" if prob > 0.5 else "ham"
    else:  # Scikit-Learn
        prob = pipeline_nb.predict_proba(msg_array)[0]
        confidence = prob[0] * 100 if prob[0] > 0.5 else prob[1] * 100
        label = "ham" if prob[0] > 0.5 else "spam"

    return f"This message is {label} at {round(confidence, 2)}% confidence"

# 7. Interface Streamlit
st.set_page_config(
    page_title="Ham or Spam?",
    page_icon="📧",
    layout="wide"
)

st.image("hamjam.png")
text_input = st.text_input('Entrez du texte :')
model_option = st.radio("Choisissez le modèle :", ('Scikit-Learn', 'TensorFlow'))

if st.button('Prédire'):
    if text_input:
        st.write(predict_message(text_input, model_option))
    else:
        st.warning("Veuillez entrer un message.")

