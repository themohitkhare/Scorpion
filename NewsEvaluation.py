import nltk
from nltk import word_tokenize
from SentimentAlgo import Evaluation

def Dictionary(News):
    token_news = word_tokenize(News)
    Diction = set(w.lower() for w in nltk.corpus.words.words())
    clean_news = ''
    for word in token_news:
        if word in Diction:
            clean_news += word + ' '
    return Evaluation(clean_news)

