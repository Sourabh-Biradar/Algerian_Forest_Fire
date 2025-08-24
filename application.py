from flask import Flask,request,render_template
import dill
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

with open('models/elcv.pkl','rb') as f:
    ml_model = dill.load(f)

with open('models/scaler.pkl','rb') as f:
    scaler = dill.load(f)

# home page
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ml_model.predict(new_data)

        return render_template('home.html',results=str(result))
    
    else :
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3838)