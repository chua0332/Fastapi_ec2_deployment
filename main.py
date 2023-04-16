import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel
import pickle


#create the app object first
app = FastAPI()

#Load both the model and the preprocessor pkl files
model_in = open('XGB_classifier.pkl','rb')
XGBmodel = pickle.load(model_in)
preprocessor_in = open('preprocessor.pkl','rb')
preprocessor = pickle.load(preprocessor_in)



#Index route, opens automatically on https://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'hello dear!'}

class fraudinput(BaseModel):
    category: object
    amt: float
    gender: object
    job: object
    merchant: object
    age: int

@app.post('/predict')
def predict_fraud(input:fraudinput):
    input_df = pd.DataFrame([input.dict().values()], columns=input.dict().keys())
    
    X_processed = preprocessor.transform(input_df)
    
    XGB_pred = XGBmodel.predict(X_processed)
    
    return {'prediction': int(XGB_pred)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



    

