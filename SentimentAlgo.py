from urllib import request, parse
import http.client, urllib.parse
import json


def Evaluation(Article):
    text = str(Article)
    print(text)
    # the input method to enter the text and parsing into required encoding
    para = {"text": text}
    data = parse.urlencode(para).encode()
    #print(data)
    # send a POST request to the sentimental analysis
    conn = http.client.HTTPConnection("text-processing.com")
    conn.request("POST", "http://text-processing.com/api/sentiment/", data)
    response = conn.getresponse()

    # Receiveing the JSON request and Displaying the results
    json_string = str(response.read())  # api/sentiment/
    #print(json_string)
    parsed_json = json.loads(json_string[2:-1])
    print("Input Text : \n " + str(text))
    print("\nEvaluation : " + parsed_json['label'])
    print("Probability : ")
    print("     Negative : " + str(parsed_json['probability']['neg']))
    print("     Positive : " + str(parsed_json['probability']['pos']))
    print("     Neutral  : " + str(parsed_json['probability']['neutral']))
    return [parsed_json['label'], str(parsed_json['probability']['pos']), str(parsed_json['probability']['neg']),
            str(parsed_json['probability']['neutral'])]
