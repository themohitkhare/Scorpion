import requests
import sqlite3
from bs4 import BeautifulSoup

dataBase = sqlite3.connect('Sting.db')

def stocklist():
    db = dataBase.cursor()
    db.execute("DROP TABLE StockCode")
    db.execute("CREATE TABLE IF NOT EXISTS StockCode(Name TEXT,BSE LONG PRIMARY KEY ,NSE LONG)")
    url = "http://www.marketonmobile.com/search.php"
    source_code = requests.get(url).text
    soup = BeautifulSoup(source_code, "html.parser")
    soup = BeautifulSoup(str(soup.find_all('div', {'class':'row'})), "html.parser")
    soup = BeautifulSoup(str(soup.find_all('ul')), "html.parser")
    temp = []
    for x in soup.find_all('li'):
        temp.append(x.string)
    print(type(temp))
    for i in range(0,len(temp),1):
        if i % 3 is 0:
            name = str(temp[i])
        elif i % 3 is 1:
            bsecode = str(temp[i])
        else:
            nsecode = str(temp[i])
            print(name + " " + bsecode + " " + nsecode)
            db.execute("INSERT INTO StockCode VALUES(?,?,?)",(name, bsecode, nsecode))
    soup = BeautifulSoup(source_code, "html.parser")
    soup = BeautifulSoup(str(soup.find_all('div', {'class': 'row ODD'})), "html.parser")
    soup = BeautifulSoup(str(soup.find_all('ul')), "html.parser")
    temp = []
    for x in soup.find_all('li'):
        temp.append(x.string)
    print(type(temp))
    for i in range(0, len(temp), 1):
        if i % 3 is 0:
            name = str(temp[i])
        elif i % 3 is 1:
            bsecode = str(temp[i])
        else:
            nsecode = str(temp[i])
            print(name + " " + bsecode + " " + nsecode)
            db.execute("INSERT INTO StockCode VALUES(?,?,?)", (name, bsecode, nsecode))
    dataBase.commit()

stocklist()
