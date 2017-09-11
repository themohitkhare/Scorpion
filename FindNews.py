import requests
import time
from bs4 import BeautifulSoup
import sqlite3
from NewsEvaluation import Dictionary

dataBase = sqlite3.connect('Sting.db')


def getnews():
    db = dataBase.cursor()
    db.execute("SELECT Name FROM Companies ORDER BY Frequency DESC")
    List = db.fetchall()
    for x in List:
        getnewsarticles(getnewslinks(x[0]),x[0])


def getnewslinks(Name):
    linklist = []
    namemodify = ""
    source_code = ''
    for x in Name:
        if (x is " "):
            namemodify += "%20"
        else:
            namemodify += x
    url = "https://news.google.com/news/search/section/q/" + namemodify + "/" + namemodify + "?hl=en-IN&ned=in"
    while source_code == '':
        try:
            source_code = requests.get(url, verify=False).text
        except:
            print("Server too slow...")
            print("Waiting for 5 sec...")
            time.sleep(5)
            print("Retrying....")
            continue

    soup = BeautifulSoup(source_code, 'html.parser')
    print(Name)
    links = soup.find_all('a', {'role': 'heading'})
    for link in links:
        linklist.append(link.get('href'))
    return linklist


def getnewsarticles(linklist, cName):
    print(cName)
    for link in linklist:
        print(link)
        source = link[:link[7:].find('/')]
        db = dataBase.cursor()
        source_code = ''
        while source_code == '':
            try:
                source_code = requests.get(link, verify = False).text
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
            if len(str(data)) >22:
                article += data
            else:
                pass
        print(article)
        print(len(article))
        if len(article) == 0:
            article = 'This is a neutral sentence'
        Eval = Dictionary(article)
        print(source)
        #print(cName)
        #print(Eval)
        db.execute('CREATE TABLE ?()')
getnews()
