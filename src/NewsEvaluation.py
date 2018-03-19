"""
Simple module to parse the given article for relevant english words and removing any other garbage strings.
"""
import nltk
from nltk import word_tokenize

from SentimentAlgo import Evaluation


def Dictionary(news):
    """
    Method tokenise the article and compares each word with present english dictionary.

    :type news: str
    :param news: News Atricle
    :return list: Evaluation object
    """
    token_news = word_tokenize(news)
    dic = set(w.lower() for w in nltk.corpus.words.words())
    clean_news = ''
    for word in token_news:
        if word in dic:
            clean_news += word + ' '
    return Evaluation(clean_news)
