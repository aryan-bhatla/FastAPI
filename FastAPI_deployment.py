# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# 1) Import libraries
import uvicorn
import pickle
import pandas as pd
import numpy as np
from BankNotes import BankNote
from fastapi import FastAPI

# 2) Create the app object
app_deployed = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


# 3) Index route, opens automatically on http://127.0.0.1:8000
@app_deployed.get("/")
def index():
    return {'message': 'Hello to whoever is reading this!'}


# 4) Route with a single parameter, returns the parameter within a message located at http://127.0.0.1:8000/<Name>
@app_deployed.get("/Welcome")
def get_name(name: str):
    return {"Welcome to Aryan's FastAPI tutorial program": f'{name}'}


# 5) Expose the prediction functionality, make a prediction
@app_deployed.post("/predict")
def predict_banknote(data: BankNote):
    data = data.dict()
    print(data)
    print("Hello")
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] > 0.5:
        prediction = "Fake Note"
    else:
        prediction = "Real Note"
    return {
        'Prediction': prediction
    }


# 6) Run the API with uvicorn on http://127.0.0.2:8000
if __name__ == '__main__':
    # Only one app can be running on one host server at a time so change from 127.0.0.1 to 127.0.0.2
    uvicorn.run(app_deployed, host='127.0.0.2', port=8000)

# 6) Run the following uvicorn command in the command prompt by first moving into the correct directory
# First parameter is file name and second parameter is name of app
# uvicorn main:app_deployed --reload

# Add /docs or /redoc to the end of http://127.0.0.2:8000 to access the /welcome section without a HTML frontend
# /docs uses SwaggerUI while /redoc uses ReDoc
