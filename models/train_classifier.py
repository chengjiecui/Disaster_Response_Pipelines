import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk import word_tokenize, pos_tag, ne_chunk, WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
    Loads the cleaned data from the database into the dataframe

    Returns:
    X: Disaster messages
    y: Disaster categories for each messages
    category_names: Message labels 
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df['message']
    Y = df.loc[:,'related':]

    category_names = Y.columns

    return X,Y,category_names

def tokenize(text):
    """
    Tokenizes the message text

    Returns a tokens list
    """
    # regex pattern for detecting a url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex,text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Creates a multi-output Random Forest nlp pipeline with grid search for performance optimization

    Returns Random Forest nlp pipeline
    """
    # 2-step pipeline for text-processing and classification
    pipeline = Pipeline([
        ('text',TfidfVectorizer(stop_words = 'english', tokenizer = tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1234, n_jobs=-1)))
    ])
    
    # set parameters for gird search
    parameters = {
        # 'text__ngram_range': [(1,1),(1,2)],
        # 'text__use_idf': (True, False),
        'clf__estimator__n_estimators': [3,5,10]
    }

    # gird search
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted', return_train_score=True, verbose=3) 

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the classification_report for predictions
    """
    
    pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        print('\033[1m'+col+'\033[0m', '\n', classification_report(Y_test.iloc[:,i], pred[:,i]))


def save_model(model, model_filepath):
    """
    Saves the model into a pickel file at model_filepath
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        print(model.best_estimator_)
        
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