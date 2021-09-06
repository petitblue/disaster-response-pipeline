import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from custom_extractor import DisasterWordExtractor


def load_data(database_filepath,table_name='cleaned_data'):
    '''
    Function to load data from sqlite database and set X and y for modeling preparation
    
    OUTPUT
        - X: array of messages
        - y: target dataframe
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, con=engine)
    # define feature and target variables X and y
    X = df['message'].values
    y = df[df.columns[4:]]
    return X, y


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def replace_url(text):
    '''
    Replace url with 'urlplaceholder' in text
    INPUT:
        text: string
    OUTPUT:
        text: edited string
    '''
    detected_urls = re.findall(url_regex, text)
    # replace each url in text strings with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    return text


def tokenize(text):
    """
        Function to processed text in string format, and output a list of words
        1. repalce all urls with 'urlplaceholder'
        2. case normalization and remove punctuation
        3. tokenize text
        4. remove stop words
        5. lemmatize words
        INPUT:
            -text: string, raw text data
        OUTPUT:
            -clean_tokens, list of processed words
    
    """

    # replace each url in text strings with placeholder
    text = replace_url(text)
    # Case Normalization
    text = text.lower() # convert to lowercase
    # remove puntuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    token_list = []
    # remove stop words
    for tok in tokens:
        if tok not in stopwords.words("english"):
             token_list.append(tok)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iritate through each token
    clean_tokens = []
    for tok in token_list:
        # lemmatize and remove leading and tailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Function that build the pipeline and the grid search parameters to
    create a classification model
    
    OUTPUT: cv:classification model
    '''
    xgb_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ])),
            ('disaster_words', DisasterWordExtractor())
        ])),
        ('clf', MultiOutputClassifier(estimator=XGBClassifier(max_depth=6)))
    ])
    # create grid search parameters
    parameters = { 'clf__estimator__n_estimators': [50,100]}
    # create grid search object
    model = GridSearchCV(xgb_pipeline, param_grid=parameters, scoring='recall_micro')
    
    return model


def evaluate_model(model, X_test, y_test):
    '''
    Function to evaluate the model for each category of the dataset
    INPUT: 
        -model: the classification model
        -X_test: the feature variable
        -y_test: the target variable

    OUTPUT:
        Classification report and accuracy score
    '''
    y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    i = 0
    for col in y_test:
        print('Feature {}:{}'.format(i+1,col))
        print(classification_report(y_test[col],y_pred[:,i]))
        i = i+1
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    '''
    The function to save machine learning pipeline 'model' to local path
    INPUT:
        model: Machine learning pipeline
        model_filepath:  the name of the local path to save the model
    OUTPUT:
        none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()