# Sentimental Ananlysis for a given text
from urllib import request, parse
import http.client, urllib.parse

import urllib
import re
def Stock():
    base_url = 'http://finance.google.com/finance/info?client=ig&q='
    exchage = "NSE"
    comp = "HDFCBANK"
    conn = http.client.HTTPConnection("finance.google.com")
    print(conn)
    conn.request("GET", base_url + exchage+"%3A"+comp)
    content = conn.getresponse()
    json_str = str(content.read()).replace("\\n","")[6:-2]
    parser_json = json.loads(json_str)
    for x in parser_json:
        print(x + " " + parser_json[x])


# from googlefinance import getQuotes
def financeinfo(symb):
    info = json.loads(json.dumps(getQuotes(symb), indent=2)[2:-2])
    for x in info:
        print(x + " : " + info[x] )

import sqlite3
import csv
dataBase = sqlite3.connect('Sting.db')

def csvconv():
    db = dataBase.cursor()
    db.execute("SELECT * FROM Companies")
    maxu = db.fetchall()
    csvWriter = csv.writer(open("Comp.csv", "w"))
    for row in maxu:
        csvWriter.writerow(row)
    print("DONE!!")    

import urllib3
import json
def getnews():
    url = "https://ajax.googleapis.com/ajax/services/search/news?"
    version = "v=1.0&"
    newstopic = input("Enter News Topic")
    newsmodify = "q="
    for x in newstopic:
        if(x is " "):
            newsmodify += "%20"
        else:
            newsmodify += x
    newsurl = url+version+newsmodify
    print(newsurl)


#getnews()






def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


install_and_import('theano')
