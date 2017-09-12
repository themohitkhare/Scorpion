import requests
import time
from bs4 import BeautifulSoup
import sqlite3
from NewsEvaluation import Dictionary

dataBase = sqlite3.connect('Sting.db')


def getnews():
    db = dataBase.cursor()
    db.execute("SELECT Name FROM Companies WHERE Frequency<250 ORDER BY Frequency DESC")
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
    links = soup.find_all('a', {'role': 'heading'})
    for link in links:
        linklist.append(link.get('href'))
    return linklist


def getnewsarticles(linklist, cName):
    print(cName)
    db = dataBase.cursor()
    Query = "CREATE TABLE IF NOT EXISTS " + str(cName).replace("&", "").replace('.', "").replace(';', "").replace("  ",
                                                                                                                  " ").replace(
        " ", "_") + "(DATE,SOURCE varchar(30),EVALUATION varchar(8),POSITIVE Float,NEGATIVE Float,NEUTRAL Float)"
    db.execute(Query)
    for link in linklist:
        source = str(link).split("//")[-1].split("/")[0] if (
        str(link).split("//")[-1].split("/")[0][:3] == 'www') else "www." + str(link).split("//")[-1].split("/")[0]
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
        if len(article) == 0:
            article = 'This is a neutral sentence'
        try:
            Eval = Dictionary(article)
            date = time.strftime("%d-%m-%Y", time.gmtime())
            print(date)
            print(source)
            print(Eval[0])
            db.execute("INSERT INTO " + str(cName).replace("&", "").replace('.', "").replace(';', "").replace("  ",
                                                                                                              " ").replace(
                " ", "_") + " VALUES (?,?,?,?,?,?)",
                       (date, source, Eval[0], Eval[1], Eval[2], Eval[3]))
        except ValueError:
            pass
    dataBase.commit()


getnews()
