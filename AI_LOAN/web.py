import pickle
import numpy as np 
from flask import Flask, render_template, request

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  

    S1 = float(request.values['SL'])
    S2 = float(request.values['SW'])
    S3 = float(request.values['PL'])
    S4 = float(request.values['PW'])

    SPECI = np.array([[S1, S2, S3, S4]]) 
    
    output = model.predict(SPECI)
    
    output = output.item()  
    
    return render_template('result.html', result="THE IRIS SPECIES IS: {}".format(output))

if __name__ == '__main__':
    app.run()
