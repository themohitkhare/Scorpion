"""
Module contains methods to collect and parse news from the internet on various companies based on the provided database.

"""
import sqlite3
import sys
import time

import requests
from bs4 import BeautifulSoup

from NewsEvaluation import Dictionary


def getnews():
    """

    Outer shell method to get news of companies present in the database.
    :return None:

    """
    dataBase = sqlite3.connect('Sting.db')
    db = dataBase.cursor()
    db.execute("SELECT Name FROM Companies ORDER BY Frequency DESC")
    List = db.fetchall()
    for x in List:
        getnewsarticles(getnewslinks(x[0]), x[0])
    dataBase.close()


def getnewslinks(Name):
    """
    Methods takes string as an input which is the name of a particular company in the indian stock market.
    Does a google news search of the comapny name and returns the list of all the news links that belong to the news
    about that comapany.

    :type Name: str
    :param Name: Name of the company
    :return linklist: list
    """

    linklist = []
    namemodify = ""
    source_code = ''
    headers = {
        'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
    for x in Name:
        if (x is " "):
            namemodify += "%20"
        else:
            namemodify += x
    url = "https://news.google.com/news/search/section/q/" + namemodify + "/" + namemodify + "?hl=en-IN&ned=in"
    while source_code == '':
        try:
            source_code = requests.get(url, headers=headers, verify=True).text
        except:
            print("Server too slow...")
            print("Waiting for 5 sec...")
            time.sleep(5)
            print("Retrying....")
            continue

    soup = BeautifulSoup(source_code, 'html.parser')
    links = soup.find_all('a', {'role': 'heading'})
    for link in links:
        linklist.append(link.get('href'))
    return linklist


def getnewsarticles(linklist, cName):
    """
    Method takes list of news links and the name of company.
    Analyses the news on each page and collect the sentiment of the entire news.
    Stores the average sentiment of the news available for each company.

    :type linklist: list
    :param linklist: List of all the hyperlinks
    :type cName: str
    :param cName: Company name
    :return None:
    """

    dataBase = sqlite3.connect('Sting.db')
    print(cName)
    LoadingCount = 0
    headers = {
        'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
    db = dataBase.cursor()
    query = "CREATE TABLE IF NOT EXISTS Company_News(DATE,NAME varchar(30) PRIMARY KEY ,EVALUATION varchar(10)" \
            ",POSITIVE Float,NEGATIVE Float,NEUTRAL Float)"

    db.execute(query)
    probability = [[], [], []]
    for link in linklist:
        sys.stdout.write("\r%d%%" % int(100 * LoadingCount / len(linklist)))
        sys.stdout.flush()
        LoadingCount += 1

        source = str(link).split("//")[-1].split("/")[0] if (
                str(link).split("//")[-1].split("/")[0][:3] == 'www') else "www." + \
                                                                           str(link).split("//")[-1].split("/")[0]
        source_code = ''
        while source_code == '':
            try:
                source_code = requests.get(link, headers=headers, verify=True).text
            except:
                print("Server too slow...")
                print("Waiting for 5 sec...")
                time.sleep(5)
                print("Retrying....")
                continue
        article = ''
        soup = BeautifulSoup(source_code, 'html.parser')
        links = soup.find_all('p')
        for articles in links:
            data = articles.string
            if len(str(data)) > 22:
                article += data
            else:
                pass
        if len(article) != 0:

            try:
                Eval = Dictionary(article)
                for x in range(3):
                    probability[x].append(float(Eval[x + 1]))



            except ValueError:
                pass
    sys.stdout.write("\r%d%% Completed\n" % 100)
    sys.stdout.flush()
    avgEval = [sum(x) / float(len(x)) for x in probability]

    fEval = "Neutral"
    if avgEval[0] > avgEval[1]:
        fEval = "Positive"
    else:
        fEval = "Negative"

    date = time.strftime("%d-%m-%Y", time.gmtime())
    db.execute("INSERT OR REPLACE INTO Company_News VALUES(?,?,?,?,?,?)",
               (date, cName, fEval, avgEval[0], avgEval[1], avgEval[2]))

    dataBase.commit()
    dataBase.close()
