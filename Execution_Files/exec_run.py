import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
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
engine = create_engine('sqlite:///disaster_response_database.db')
df = pd.read_sql_table('disaster_messages_table', engine)

# load model
model = joblib.load('classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    related_counts = df['related'].value_counts()
    related_names = ['related', 'not related']
    
    weather_counts = list(df[df.columns[-8:-1]].sum())
    weather_names = ['Weather Related', 'Floods', 'Storm', 'Fire', 'Earthquake', 'Cold', 'Other Weather']
    
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # start graph genre
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
        }, # end graph genre
        # start graph related
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related"
                }
            }
        }, # end graph related
        # start graph weather
        {
            'data': [
                Bar(
                    x=weather_names,
                    y=weather_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Weather Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Weather"
                }
            }
        } # end graph weather
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