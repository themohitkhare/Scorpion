Scorpion
======================
Sorpion is a indian stock market predictor that uses LSTM and Sentiment Analysis.

## Description
Scorpion was created  out of the fact that the stock values of companies change not only based on there quaterly reports
and productions but majorly depends on the their image and how they perform in the public eye. A great example is the
company Tesla which has a market valuation of 45 billion  as comapred to GM(General Motors) which has a market valuation
of 40 billions. But Tesla produces around 4% of the cars as compared to GM.



## Requirements 
**Update:** As of 28-02-2018 this code works with Python3.5

* Python 3.5
* BeautifulSoups 0.0.1
* Tensorflow/Tensorflow-gpu 1.4.0(or Theano)
* Keras 2.1.4
* Matplotlib 2.1.2
* Numpy 1.14.1
* Pandas 0.22.0
* Quandl 3.3.0
* Sklearn 0.0
* NLTK 3.2.5

## Usage Instructions
Install the required packages before running the program.
    
    pip install -r requirements.txt

Start the program by running main.py. It is a menu driven program for easy execution.
    
    python3 main.py


**Note:** 
* For using tensorflow-gpu installation of required CUDA libraries and CNN is required.
* Download nltk corpus (words)
    ```
    >>> import nltk
    >>> nltk.download()
    ```

##Results
   **HDFC Bank**
   ![alt text](https://github.com/mohitkhare582/Scorpion/tree/master/Graphs\HDFC Bank.png "HDFC Bank")
   
   
   **ICICI Bank**
   ![alt text](https://github.com/mohitkhare582/Scorpion/tree/master/Graphs\ICICI Bank.png "ICICI Bank")
   
   
   **TATA Steel**
   ![alt text](https://github.com/mohitkhare582/Scorpion/tree/master/Graphs\TATA Steel.png "TATA Steel")
   
  
    
    

## Future
- Upcoming updates
    * Prediction for particular company.
    * Prediction for fucture days.
    * Incresing accuracy for news evaluation.

Support
-------
Please file bugs and issues on the Github issues page for this project. This is to help keep track and document everything related to this repo. For general discussions and further support you can create a pull request. The code and documentation are released with no warranties or SLAs and are intended to be supported through a community driven process.
