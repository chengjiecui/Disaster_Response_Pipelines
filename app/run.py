import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disaster_Response.db')
df = pd.read_sql_table('Disaster_Response', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # message length by Character
    message_lengths = df.message.str.len()
    # 10 bins counts of message length within 4th quartile
    length_percents=(message_lengths[message_lengths<=message_lengths.quantile(.99)].value_counts(normalize=True,bins=10).round(decimals=4).sort_index())*100
    # generate x-ticks base on Serise.value_counts.index for plotly
    xticks = []
    for idx in length_percents.index:
        xticks.append(str(int(idx.left))+'-'+str(int(idx.right)))
    # correlation between labels
    labels = df.loc[:,'related':]
    cor = labels.corr()
    

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=xticks,
                    y=length_percents.values
                    )
            ],

            'layout': {
                'title': 'Message Length by Character',
                
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=cor.values, 
                    x=cor.columns,
                    y=cor.index
                )
            ],

            'layout': {
                'title': 'Heatmap of Labels',
                'height': 1100
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()