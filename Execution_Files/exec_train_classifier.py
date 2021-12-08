# import libraries
from sqlalchemy import create_engine

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import statements
import sys
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#import ml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"


def load_data(database_filepath):
    """
    Loads data from the database and defines variables for the upcoming part
    
    Parameters:
    - Filepath where database is stored
    
    Returns:
    - A Pandas Series with column 'messages' from the dataframe
    - A dataframe with the 36 'categories' columns
    - A list with the names of the 36 'categories' columns
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name="disaster_messages_table", con='sqlite:///' + database_filepath)

    X = df['message']
    Y = df.drop(columns=['id','message', 'original', 'genre'])
    col_names = list(Y.columns)
    return X, Y, col_names


def tokenize(text):
    """
    Processes text data
    
    Parameters:
    - Text data, in this case entries from the Pandas Series 'messages'
    
    Returns:
    - A list with words of the input sentence
    """
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Builds the machine learning pipeline which takes in the message column as input and outputs classification results on the other 36
    categories in the dataset
    
    Parameters:
    - None
    
    Returns:
    - The built machine learning pipeline
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_features=5000)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 50],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters , verbose=3)
    
    return cv


def evaluate_model(pipeline, X_test, Y_test, category_names):
    """
    Reports the f1 score, precision and recall for each output category of the dataset.
    
    Parameters:
    - The built machine learning pipeline
    - The two test-DataFrames from the train-test-split
    - A list with words of the input sentence
    
    Returns:
    - None (The classification_report is only printed)
    """

    Y_pred = pipeline.predict(X_test)
    Y_test_np = Y_test.values
    print(classification_report(Y_test_np, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the model (ML pipeline) into a pickle file
    
    Parameters:
    - The model (ML pipeline)
    - Filepath where pickle file should be stored
    
    Returns:
    - None
    """
        
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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