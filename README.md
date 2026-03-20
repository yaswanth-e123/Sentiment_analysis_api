Sentiment Analysis

This project builds a Sentiment Analysis system that classifies text as Positive or Negative using Machine Learning.
It includes data preprocessing, model training, evaluation, and an API for real-time predictions.

Features
Text preprocessing (cleaning, normalization)
TF-IDF feature extraction
Logistic Regression model
Model evaluation (Accuracy, Classification Report, Confusion Matrix)
REST API using FastAPI
Real-time prediction endpoint

Python
scikit-learn
Pandas
FastAPI
Joblib

the gievn dataset by techminds has

The dataset must contain 2 columns:
text --- input text
label --- sentiment (positive / negative)

Example
text,label
"I love this techminds",positive
"This is terrible",negative

installing

create venv:

python -m venv venv

activate environment

venv\Scripts\activate

install the requirements.txt
python -m pip install -r requirements.txt

Model run
python model/train.py

These gives these as outputs:
Accuracy
Classification Report (Precision, Recall, F1-score)
Confusion Matrix



FastAPI server
uvicorn app.main:app --reload
Endpoint
POST /predict
These give as input
json
{
  "text": "I love this product"
}

json
{
  "input_text": "I love this product",
  "cleaned_text": "i love this product",
  "prediction": "positive"
}

In nlp what i am done
Lower case
Removed numbers(notneeded our textnot have number but it is better to do)
Remove punctuation
Removed extra spaces
Removing stopwords( via TF-IDF)

Model taken TF-IDF vectorization and logestic regression

Here mymodel is not performing well
Reasons:
Dataset size is to low
If have  any class more than other imbalance created


to Perform well you can use 
TF_IDF place contex_vector 
ADD more data to the dataset
makesure the class should be balance
make sure use other models better than logstic regression we have decision tree classifier,adaboost classifier



