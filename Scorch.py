"""
Scorpion
This is a python program that uses data mining and advance sentimental analysis to provide the list of companies in the
stock market.


"""

import requests
import time
import urllib3.exceptions
from bs4 import BeautifulSoup
import sqlite3

dataBase = sqlite3.connect('Sting.db')

# Function to check the availability of a fund Code
def checkfund(code):
    url = "https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode=" + str(code)
    if requests.head(url).status_code == 200:
        return True
    else:
        return False


def getfundinfo(code):
    link = "https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode=" + str(code)
    source_code = ''
    while source_code == '':
        try:
            source_code = requests.get(link).text
        except:
            print("Server too slow...")
            print("Waiting for 5 sec...")
            time.sleep(5)
            print("Retrying....")
            continue
    soup = BeautifulSoup(source_code, "html.parser")
    # This code will fetch the fund type
    table = soup.find('td', {'width': '240px'})
    colon = table.string.index(":")
    fundType = table.string[:colon]
    if "Debt" == fundType or "Debt" in table.string:
        print("Bad Code " + str(code))
        pass
    else:
        # This code will fetch the fund name
        for url in soup.find_all('span', {'class': 'fundname_rating_sep'}):
            FundName = url.string
        if (len(FundName) == 0):
            for urlo in soup.find_all('h1', {'class': 'pull-left fund-name'}):
                FundName = urlo.string

        # This code will fetch the list of companies in the particular fund
        cstr = " "
        table = soup.find('table', {'id': 'fund-snapshot-port-holdings'})
        rows = table.find_all('tr')
        # Will Print all the Government Companies
        for row in rows:
            for col in row.find_all('td'):
                for company in col.find_all('a'):
                    stock_code = company.get('href')
                    if 'code' in stock_code:
                        cstr += (company.string + ", ")

        print(str(code) + " " + str(FundName) + " " + fundType + " " + cstr)
        db = dataBase.cursor()
        db.execute("CREATE TABLE IF NOT EXISTS Funds(Code LONG PRIMARY KEY ,Fund_Type TEXT, Fund_Name TEXT, Company_List TEXT)")
        db.execute("INSERT INTO Funds VALUES(?,?,?,?)", (code, fundType, FundName, cstr))
        dataBase.commit()


# This Function will loop and create the Funds database
def createdatabase():
    db = dataBase.cursor()
    db.execute('DELETE FROM Funds Where Company_List is " "')
    db.execute("SELECT MAX(Code) FROM Funds")
    maxcode = db.fetchall()
    ilist = []
    maxcode = str(maxcode)
    for x in maxcode:
        if x.isnumeric():
            ilist.append(x)
    i = ''.join(ilist)
    print(i)
    i = int(i) + 1
    while True:
        if checkfund(i):
            getfundinfo(i)
            i += 1
        else:
            print("Bad Code " + str(i))
            i += 1


# This Function will create the database of the Companies of the Funds
def createcompanydb():
    db = dataBase.cursor()
    db.execute("DROP TABlE Companies")
    db.execute("SELECT Company_list FROM FUNDS")
    clist = db.fetchall()
    comlistall = []
    liststring = ""
    for x in clist:
        liststring = x[0]
        for i in liststring.split(','):
            comlistall.append(i[1:])
    compdict = {}
    for company in comlistall:
        if company in compdict:
            compdict[company] += 1
        else:
            compdict[company] = 1
    db.execute("CREATE TABLE IF NOT EXISTS Companies(Name TEXT,Frequency LONG)")
    for value in compdict:
        db.execute("INSERT INTO Companies VALUES(?,?)",(value, compdict[value]))
    db.execute("DELETE FROM Companies WHERE Name is ''")
    dataBase.commit()


#This function will Create the list of all the Stocks that are in Indian Stock Market
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

# This is the Main Function that will work as the menu for the whole code.
def main_menu():
    menu = {'1': "Create Database.", '2': "Update Company Database", '3': "Find Student", '4': "Exit"}
    while True:
        for key in sorted(menu.keys()):
            print("%s: %s" % (key, menu[key]))
        selection = input("Please Select:")
        if selection == '1':
            print("Creating DataBase")
            createdatabase()
        elif selection == '2':
            print("Updating Company Database")
            createcompanydb()
            print("Update Complete")

        elif selection == '3':
            print("Update Stock Code List")
            stocklist()
            print("Stock List Updated")
        elif selection == '4':
            break
        else:
            print("Unknown Option Selected!")


main_menu()
