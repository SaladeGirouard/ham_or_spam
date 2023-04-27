
import streamlit as st

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def firstco():
    import pandas as pd
    import nltk
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import string
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    # deep learning libraries for text pre-processing
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # Modeling 
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
    
    dfspam = pd.read_csv('SMSSpamCollection.csv',sep="	")

    # Définir les stopwords, la ponctuation et le lemmatiseur
    stopwords = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()

    # Fonction pour nettoyer le texte
    def clean_text(text):
        # Mettre en minuscule
        text = text.lower()
        # Enlever la ponctuation
        text = ''.join(char for char in text if char not in punctuation)
        # Enlever les stopwords et lemmatiser
        tokens = nltk.word_tokenize(text)
        text = ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stopwords)
        return text
    
    dfspam['comment_clean'] =0
    for i in range(len(dfspam['comment'])):
        dfspam['comment_clean'].iloc[i] = clean_text(dfspam['comment'].iloc[i])

    X = dfspam['comment_clean']
    y = dfspam['Type']

    def classify(model, X, y):
        # train test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
        # model training
        pipeline_model = Pipeline([('vect', CountVectorizer()),
                                ('tfidf',TfidfTransformer()),
                                ('clf', model)])
        pipeline_model.fit(x_train, y_train)
        
        y_pred = pipeline_model.predict(x_test)
        
        return pipeline_model

    model_naive = MultinomialNB()

    pipeline_model = classify(model_naive, X, y)

    # Get all the ham and spam emails
    ham_msg = dfspam[dfspam.Type =='ham']
    spam_msg = dfspam[dfspam.Type=='spam']

    dfspam['msg_type']= dfspam['Type'].map({'ham': 0, 'spam': 1})
    msg_label = dfspam['msg_type'].values

    train_msg, test_msg, train_labels, test_labels = train_test_split(dfspam['comment_clean'], msg_label, test_size=0.2, random_state=22)

    # Defining pre-processing hyperparameters
    max_len = 50 
    trunc_type = "post" 
    padding_type = "post" 
    oov_tok = "<OOV>" 
    vocab_size = 500

    tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
    tokenizer.fit_on_texts(train_msg)

    # Sequencing and padding on training and testing 
    training_sequences = tokenizer.texts_to_sequences(train_msg)
    training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
    testing_sequences = tokenizer.texts_to_sequences(test_msg)
    testing_padded = pad_sequences(testing_sequences, maxlen = max_len,
    padding = padding_type, truncating = trunc_type)

    embeding_dim = 16
    drop_value = 0.2 # dropout
    n_dense = 24

    #Dense model architecture
    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(drop_value))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])

    # fitting a dense spam detector model
    num_epochs = 30
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels),callbacks =[early_stop], verbose=2)

    return dfspam, model, pipeline_model


### CONFIGURATION DE LA PAGE ###
st.set_page_config(
     page_title="Ham or spam ?",
     page_icon="📧",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
        'Get Help':  None,
         'Report a bug': None,
         'About': "# Bienvenue ! # \n"
         "Un projet de Romain Ulrich, Adil Allouche et Charles Girouard"
     }
)

### VARIABLES ###
dfspam, model, pipeline_model = firstco()

def Spam_or_ham(msg :str, model: str):
    msg = clean_text(msg)
    msgarray = []
    msgarray.append(msg)
    msgarray = np.array(msgarray)
    if model == "TensorFlow":
        sequences = tokenizer.texts_to_sequences(msgarray)
        padded = pad_sequences (sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
        result = model.predict(padded,
        batch_size=None,
        verbose='auto',
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False)
        if result[0][0]<0.2 :
            confidence = (1-result[0][0])*100
            text = 'This message is a ham at '+str(round(confidence,2))+' % confidence'
        else:
            confidence =result[0][0]*100
            text = 'This message is a spam at '+str(round(confidence,2))+' % confidence'
    else :
        prediction = pipeline_model.predict(msgarray)
        if prediction[0] == "ham":
            text= 'This message is a ham'
        else:
            text='This message is a spam'
    return text


##### INTERFACE #####

st.write(
    "<div style='text-align: center;'><h1 style='font-family: Comic Sans MS;'>Ham or Spam?</h1></div>",
    unsafe_allow_html=True
)


text_input = st.text_input('Entrez du texte :', key='input_text')


option = st.radio('On part sur quel algo ?', ('Scikit-Learn', 'TensorFlow'))


if option == 'Scikit-Learn':
    st.session_state['option2_checked'] = False
else:
    st.session_state['option1_checked'] = False



if st.button('Valider'):
    if text_input:
        Spam_or_ham(text_input, option)
    else:
        st.write("Veuillez entrer du texte dans la zone de texte.")


