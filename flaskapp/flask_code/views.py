from flask import render_template
from flask import request
from flask_code import app
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2
from model_prediction import make_prediction

model_path = '../trained_models/common_combo.pt'

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/go')
def go():
    query = request.args.get('query', '')
    
    predictions, probs = make_prediction(model_path,query)
    
    return render_template(
        'go.html',
        predictions=predictions,
    )

