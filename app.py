import pickle
import re, string
import streamlit as st
import webbrowser
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

detectFile = open('model.pkl','rb')
detectModel = pickle.load(detectFile)
detectFile.close()
st.title("Movie Fake Review Detection Tool")
input_test = st.text_input("Provide text input here",placeholder="Enter comment here")

# Removing all punctuations from review
mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "don't" : "do not"}

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(review):
    return review.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def clean_contractions(review, mapping):
    specials = ["’", "‘", "´", "`", "_"]
    for s in specials:
        if s == "_":
          review = review.replace(s, " ")
        else:
          review = review.replace(s, "'")
    review = ' '.join([mapping[t] if t in mapping else t for t in review.split(" ")])
    return review


from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(review):
    return " ".join([word for word in str(review).split() if word not in STOPWORDS])

def word_replace(review):
    return review.replace('<br />','')


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(review):
    return " ".join([stemmer.stem(word) for word in review.split()])


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(review):
    return " ".join([lemmatizer.lemmatize(word) for word in review.split()])


def remove_urls(review):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', review)


def remove_html(review):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', review)


def preprocess(review):
    review=clean_contractions(review,mapping)
    review=review.lower()
    review=word_replace(review)
    review=remove_urls(review)
    review=remove_html(review)
    review=remove_stopwords(review)
    review=remove_punctuation(review)
    # review=stem_words(review)
    review=lemmatize_words(review)
    
    return review

button_clicked = st.button("Predict")
if button_clicked:
    final = preprocess(input_test)
    res = detectModel.predict([final])
    if res==0:
        st.text("The review is Fake")
    else:
        st.text("The review is genuine")