"""
Driver module for the entire project.
"""
from Scorch import *
from NeuralNetAnalysis import StockPrediction
from FindNews import getnews


if __name__ == '__main__':
    """
    Main method displays a menu based program for running different modules.
    :return: None 
    """
    menu = {'1': "Create Database.", '2': "Update Company Database", '3': "Update Stock Code List", "4": "Collect News",
            '5': "Start Stock Prediction", '6': "Exit"}
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
            print("Updating Stock Code List")
            stocklist()
            print("Stock List Updated")
        elif selection == '4':
            print("Collecting News")
            getnews()
            print("News Collected")
        elif selection == '5':
            print("Predicting Stocks")
            StockPrediction()
            print("Stock Predicted")
            break
        else:
            print("Unknown Option Selected!")


