import sys, re, string
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Download used NLTK assets
nltk.download(['punkt', 'wordnet', 'stopwords'])


##########  ##########  ##########  ##########
## Custom functions
def display_metrics_multioutput(y_test, y_pred, metrics_dec_places=3):
    '''
    Parameters
    ----------
    y_test : list(numeric)
        True labels of the test data set
    y_pred : list(numeric)
        Predicted labels on the test data set
    metrics_dec_places : int, optional
        Numbers of decimal places the result should be rounded (default is 3)

    Return
    ------
    df_metrics : Pandas DataFrame
        A table showing model performance metrics (accuracy, precision, recall, f1-score)
        for all given prediction columns
    '''
    # Prepare Classification Report DataFrame
    df_metrics = pd.DataFrame(columns=['feature_name', 'accuracy', 'precision', 'recall', 'f1-score'])

    # iterate over all classification columns, add model metrics
    for i, col in enumerate(y_test):
        df_metrics.loc[len(df_metrics)] = [col, \
                                           accuracy_score(y_test[col], y_pred[:, i]), \
                                           precision_score(y_test[col], y_pred[:, i], average='weighted'), \
                                           recall_score(y_test[col], y_pred[:, i], average='weighted'), \
                                           f1_score(y_test[col], y_pred[:, i], average='weighted')]

    df_metrics.set_index('feature_name', inplace=True)

    return df_metrics.round(metrics_dec_places)


##########  ##########  ##########  ##########
## Predefined functions

def load_data(database_filepath):
    '''
    Parameters
    ----------
    database_filepath : str
        Database file name which will be loaded

    Return
    ------
    X : list object
        Contains all text messages
    y : Pandas DataFrame
        All categories as response variable for X
    category_names : list
        Names of all categories, stored in y
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('CategorizedMessages', con=engine)
    X = df['message']
    y = df.drop(['id','message','original', 'genre'], axis=1)
    category_names = y.columns

    return X, y, category_names



def tokenize(text, stop_word_corpus='english'):
    '''
    Parameters
    ----------
    text : str
        Text to be preprocessed
    stop_word_corpus [optional] : str
        Corpus for the stop word identification and removal.


    Return
    ------
    clean_tokens : tokens
        Preprocessed and tokenized input text: removed stopwords, lemmatized and tokenized.
    '''
    nlp_stop_words = stopwords.words(stop_word_corpus)
    nlp_lemmatizer = WordNetLemmatizer()
    # nlp_remove_punctuation = RegexpTokenizer(r'\w+') # apply: nlp_remove_punctuation.tokenize('my, text!')
    nlp_remove_punctuation = str.maketrans('', '', string.punctuation)

    # 1) Normalize (remove punction, all cases to lower)
    text_prep = text.translate(nlp_remove_punctuation).lower()
    # 2) Tokenize as words
    tokens = word_tokenize(text_prep)
    # 3) Lemmatize
    clean_tokens = [nlp_lemmatizer.lemmatize(w) for w in tokens if not w in nlp_stop_words]

    return clean_tokens



def build_model():
    '''
    Parameters
    ----------
    /


    Return
    ------
    model_pipeline : scikit-learn Pipeline
        Configured scikit-learn Pipeline, prepared for fitting data
    '''
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf_rnd', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])

    # define parameters for GridSearchCV --> the others are deactived due to very high computation time. This is just to show how it works...
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True],
        'clf_rnd__estimator__min_samples_split': [2, 4]
        # 'vect__max_df': (0.5, 1.0),
        # 'vect__max_features': (None, 10000),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'clf_rnd__estimator__n_estimators': [50, 100, 200, 500],
        # 'clf_rnd__estimator__max_depth': [4, 5, 6, 7, 8],
        # 'clf_rnd__estimator__criterion': ['gini', 'entropy']
    }

    # create gridsearch object and return as final model pipeline
    # model_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, n_jobs=-1) # n_jobs=-1 --> This will parallelize and speed up the G.S.CV; My machine had issues, thus I deactived it!
    model_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3)

    return model_pipeline


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Parameters
    ----------
    model : classifier
        Classification model
    X_test :
        Text messages from the test data set
    y_test :
        Response variables from the test data set
    category_names : list
        Column names of y_test
    Return
    ------
    df_result : Pandas DataFrame
        Provides information about all scored response variables.
    '''
    y_pred = model.predict(X_test)

    df_result = display_metrics_multioutput(y_test, y_pred)
    print('Accuracy of complete model:', (y_pred == y_test).mean().mean())

    return df_result


def save_model(model, model_filepath):
    '''
    Parameters
    ----------
    model : classifier
        Classification model
    model_filepath : str
        Name of the file

    Return
    ------
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        df_result = evaluate_model(model, X_test, y_test, category_names)
        print(df_result)

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