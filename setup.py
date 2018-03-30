from distutils.core import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='Scorpion',
      version='1.0',
      description='Stock Market Prediction',
      author='Mohit Khare',
      author_email='mohitkhare582@gmail.com',
      packages=['sys', 'time', 'sqlite3', 'bs4', 'requests', 'tensorflow', 'keras', 'matplotlib', 'numpy', 'pandas',
                'quandl', 'sklearn', 'nltk', 'json', 'urllib', 'nltk.word_tokenize', 'nltk.corpus.words', 'tensorflow-gpu', 'theano'],
      long_description=read('README.md'),
      )
