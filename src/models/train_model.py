# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
import json
import shutil
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from classes.model import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
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
    datafile_listing = glob.glob( os.path.join(input_filepath, '*.csv'))
    df = pd.DataFrame()

    for filename in datafile_listing:
        queue_df = pd.read_csv(filename)
        df = df.append(queue_df)

    return df


def get_models(file_path):
    """
    Retrieves all model metadata and re-instantiates all model objects from file.
    """
    # if `metadata.json` does not exist, initialize it:
    if not os.path.isfile(os.path.join(file_path, "models.json")):
        with open(os.path.join(file_path, "metadata.json"), "w") as file:
            file.write('{"models": []}')

    with open(os.path.join(file_path, "metadata.json"), "r") as file:
        metadata = json.load(file)

    resulting_models = []
    for model in metadata["models"]:
        with open(os.path.join(file_path, model['file_name']), "rb") as file:
            model_object = pickle.load(file)
        resulting_models.append(
            Model(model.file_name,
                  model_object,
                  model.area_under_curve
                  ))

    return resulting_models


def get_model_auc(X_test, y_test, clf):
    """
    Calculates the Area Under the Curve for a model.
    """
    print(len(X_test))
    print(len(y_test))

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

    # Update area under the curve for new models
    for model in models:
        model.area_under_curve = get_model_auc(X_test, y_test, model)

    best_model = max(model, key=lambda model: model.area_under_curve)

    return best_model


def move_to_production(output_filepath, model):
    """Moves a model binary into production."""

    source = os.path.join(output_filepath, model.file_name)
    destination = os.path.join(output_filepath, 'production', model.file_name)

    # delete contents
    for file in os.scandir(destination):
        if file.name.endswith(".pkl"):
            os.unlink(file.path)

    shutil.copyfile(source, destination)


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    This module trains the Random Forest Classifier model if it does not yet exist,
    or if it does exist updates the model and selects the best performing one for production.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

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
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

    classifier_model_plot(X_test, y_test, clf, (20, 10), output_filepath, f'model-{date_time}-evaluation-plots.png')

    # Save to file
    pkl_filename = f"model-{date_time}.pkl"
    with open(os.path.join(output_filepath, pkl_filename), 'wb') as file:
        pickle.dump(clf, file)

    # Add to model collection and compare for best model
    # TODO: functies get_models() en compare_models() methodes maken van Model class?
    models = get_models(output_filepath)

    # TODO: gaat hier alles goed als models = [] ?
    clf_model = Model(pkl_filename, clf, get_model_auc(X_test, y_train, clf))
    clf_model.add_to_metadata(output_filepath)

    if len(models) > 0:
        models.append(clf_model)
        best_model = compare_models(models, clf, data, X_test, y_test)
    else:
        best_model = clf_model

    if best_model.filename == pkl_filename:
        logger.info(f'Currently trained model {pkl_filename} was deemed best model! Moved model to production.')
    else:
        logger.info(f'Earlier trained model {best_model.file_name} was deemed best model. Moved model to production.')

    print(best_model)

    move_to_production(output_filepath, best_model)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    print(project_dir)
    inf = os.path.join(project_dir, "data", "processed")
    outf = os.path.join(project_dir, "models")
    main(inf, outf)
