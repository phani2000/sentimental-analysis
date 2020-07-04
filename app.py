from flask import Flask, render_template, request
from flask import *
import nltk

import NaiveBayes as s1

nltk.download('stopwords')

app = Flask(__name__)


@app.route('/')
def openpage():
	return render_template("MainPage.html")


#@app.route('/submit_details', methods=['POST', 'GET'])
@app.route('/submit_details', methods=['POST'])
def login():
    if request.method == 'POST':
        data = request.form.to_dict()
        print(f'The info in data : {data}')
        rece_data = s1.output(data['comment'])
        if rece_data == 'Positive Response':
                return render_template('Positive.html',
                                       getinfo=rece_data)

        elif rece_data == "INVALID RESPONSE" :
                return render_template('Invalid.html',
                                       getinfo=rece_data)

        else:
                return render_template('Negative.html',
                                       getinfo=rece_data)

@app.route('/<string:page_name>')
def multiplepages(page_name):
    return render_template(page_name)


if __name__ == "__main__":
    #app.run(debug=False)
    #app.run(debug=True)
    app.run()
