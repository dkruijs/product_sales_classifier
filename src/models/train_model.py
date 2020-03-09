# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from .model import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix


def classifier_model_plot(X_test, y_test, clf, figsize, output_filepath, name):
    """
    Create several plots to evaluate model performance
    """

    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]
    y_pred_labels = clf.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds2 = precision_recall_curve(y_test, y_pred)

    feats = {}  # a dict to hold feature_name: feature_importance

    for feature, importance in zip(X_test.columns, clf.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances['sd'] = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    lw = 2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=figsize)

    ax1.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.axis(xmin=0, xmax=1.05, ymin=0, ymax=1.05)
    ax1.plot([-0.05, 1.05], [-0.05, 1.05], color='navy', lw=lw, linestyle='--')
    ax1.axis(xmin=0, xmax=1, ymin=0, ymax=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Model ROC')
    ax1.legend(loc="lower right")

    confusion_matrix = plot_confusion_matrix(clf, X_test, y_test, normalize='pred', ax=ax2)
    ax2.set_title('Confusion Matrix')

    ax3.plot(precision, recall, color='darkorange',
             lw=lw, label='Precision Recall Curve')
    ax3.axis(xmin=0, xmax=1.05, ymin=0, ymax=1.05)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Recall')
    ax3.set_title('Model PRC')
    ax3.legend(loc="lower right")

    textstr = '\n'.join((
        r'$Accuracy=%.2f$' % (accuracy_score(y_test, y_pred_labels),),
        r'$Precision=%.2f$' % (precision_score(y_test, y_pred_labels),),
        r'$Recall=%.2f$' % (recall_score(y_test, y_pred_labels),)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax3.text(0.10, 0.15, textstr, transform=ax3.transAxes, fontsize=14, bbox=props)

    importances.sort_values(by='Gini-importance').plot.bar(yerr='sd', rot=90, ax=ax4)
    ax4.set_title('Importance of Variables')

    fig.savefig(os.path.join(output_filepath, name))


# TODO omzetten naar het zoeken en concateneren van alle nieuw aangeleverde data files
def get_dataset(input_filepath):
    """
    Retrieves all data files and creates a master dataset
    """
    historical = pd.read_csv(input_filepath)
    return historical


def get_models(file_path="../models"):
    """
    Retrieves all model metadata and re-instantiates all model objects from file.
    """
    with open(os.path.join(file_path, "metadata.json"), "r") as file:
        metadata = json.read(file)

    models = []
    for metadatum in metadata:
        with open(os.path.join(file_path, metadatum.file_name), "rb") as file:
            model_object = pickle.load(file)
        models.append(
            Model(metadatum.file_name,
                  model_object,
                  metadatum.area_under_curve
                  ))

    return models


def get_model_auc(X_test, y_test, clf):
    """
    Calculates the Area Under the Curve for a model.
    """
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    fpr, tpr = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def compare_models(models, clf, data, X_test, y_test):
    """
    Compares AUC scores for all pre-existing models, and the new one, and 
    returns the best performing model.

    models:
        List of previously created Model instances, found in the 'models' folder.
    clf:
        The newest classifier model.

    returns:
        The best performing model.
    """

    # Updates area under the curve for new models
    for model in models:
        model.area_under_curve = get_model_auc(X_test, y_test, model)

    best_model = max(model, key=lambda model: model.area_under_curve)

    clf_auc = get_model_auc(X_test, y_test, clf)

    return None


def move_to_production(output_filepath, model):
    """Moves a model binary into production."""

    source = os.path.join(output_filepath, model.file_name)
    destination = os.path.join(output_filepath, 'production', model.file_name)

    # delete contents
    for file in os.scandir(destination):
        if file.name.endswith(".pkl"):
            os.unlink(file.path)

    shutil.copyfile(source, destination)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    This module trains the Random Forest Classifier model if it does not yet exist,
    or if it does exist updates the model and selects the best performing one for production.
    """

    # Train a new version of the model 
    data = get_dataset(input_filepath)

    x_cols = ['ReleaseNumber', 'New_Release_Flag', 'StrengthFactor',
              'PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice',
              'MarketingTypeD']

    y_col = 'SoldFlag'

    X_train, X_test, y_train, y_test = train_test_split(data[x_cols],
                                                        data[y_col].values,
                                                        test_size=0.30,
                                                        random_state=42)

    clf = RandomForestClassifier(max_depth=5,
                                 random_state=42,
                                 criterion='gini',
                                 n_estimators=100,
                                 verbose=1,
                                 class_weight='balanced')

    clf.fit(X_train, y_train)

    # Evaluate model
    # TODO versioning van plots
    classifier_model_plot(X_test, y_test, clf, (20, 10), output_filepath, 'Model_evaluation_plots.png')

    # Save to file
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    pkl_filename = f"model-{date_time}.pkl"
    with open(os.path.join(output_filepath, pkl_filename), 'wb') as file:
        pickle.dump(clf, file)

    # Add to model collection and compare for best model
    with get_models() as models:

        clf_model = Model(pkl_filename, clf, get_model_auc(clf))
        clf_model.add_to_metadata()

        if len(models) > 0:
            models.append(clf_model)
            best_model = compare_models(models, clf, data, X_test, y_test)
        else:
            best_model = clf_model

    move_to_production(output_filepath, best_model)
