# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
from pathlib import Path
from flask import Flask, jsonify, request

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@app.route('/', methods=["POST"])
def main(model_filepath):
    """
    This module serves model predictions and saves data and result logs.
    """
    # Check if classifier is alread loaded:
    if clf is None:
        with open(model_filepath, "rb") as file:
            clf = pickle.load(file)

    request_data = pd.DataFrame(request.get_json())

    result = clf.predict(request_data)

    return jsonify(result)

