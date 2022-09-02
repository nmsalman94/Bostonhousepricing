import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app= Flask(__name__)  #starting point
##Load the model
model=pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/') #route the page

def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])  ## sending a request

def predict_api():
    data=request.json['data']  ## storing the information from request
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__== "__main__":
    app.run(debug=True)


