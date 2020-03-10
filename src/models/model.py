import os
import json
import pickle
import shutil
from pathlib import Path

from sklearn.metrics import roc_curve, auc, precision_recall_curve

def get_model_auc(X_test, y_test, clf):
    """
    Calculates the Area Under the Curve for a model.
    """
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def compare_models(models, X_test, y_test):
    """
    Compares AUC scores for all pre-existing models, and the new one, and
    returns the best performing model.

    models:
        List of previously created Model instances, found in the 'models' folder.
    X_test:
        Pandas dataframe with independent variable columns of the test dataset
    y_test:
        Pandas dataframe with dependent variable columns of the test dataset

    returns:
        The model with the highest area under the curve.
    """

    # Update area under the curve for new models
    for model in models:
        model.area_under_curve = get_model_auc(X_test, y_test, model.model_object)

    from operator import attrgetter
    best_model = max(models, key=attrgetter('area_under_curve'))

    return best_model


def move_to_production(output_filepath, model):
    """Moves a model binary into production."""

    source = os.path.join(output_filepath, model.file_name)
    destination_dir = os.path.join(output_filepath, 'production')
    destination_file = os.path.join(output_filepath, 'production', model.file_name)

    # delete contents
    for file in os.scandir(destination_dir):
        if file.name.endswith(".pkl"):
            os.unlink(file.path)

    shutil.copyfile(source, destination_file)


def initialize_metadata(file_path):
    """Initializes the model metadata file `metadata.json`."""
    with open(file_path, "wt") as file:
        file.write('{"models": []}')


def get_models(file_path):
    """
    Retrieve all model metadata and re-instantiates all model objects from file.
    Return an empty list if no models are found.
    """
    # if `metadata.json` does not exist, initialize it:
    metadata_path = os.path.join(file_path, "metadata.json")

    if not os.path.isfile(metadata_path):
        initialize_metadata(metadata_path)

    # load metadata
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    resulting_models = []
    for model in metadata["models"]:
        with open(os.path.join(file_path, model['file_name']), "rb") as model_file:
            model_object = pickle.load(model_file)
        resulting_models.append(
            Model(model['file_name'],
                  model_object,
                  model['area_under_curve'],
                  model['evaluation_plot_file_name']
                  ))

    return resulting_models


class Model(object):
    """The Model class collects a model binary with model metadata."""

    def __init__(self, file_name, model_object, area_under_curve, evaluation_plot_file_name):
        self.file_name = file_name
        self.model_object = model_object
        self.area_under_curve = area_under_curve
        self.evaluation_plot_file_name = evaluation_plot_file_name

    def __repr__(self):

        return f"""
            file_name: {self.file_name}
            model_object: {self.model_object}
            area_under_curve: {self.area_under_curve}
            evaluation_plot_file: {self.evaluation_plot_file_name}
        """

    def add_to_metadata(self, file_path):
        """Adds a model to the metadata.json file."""
        path = os.path.join(file_path, "metadata.json")

        # check if metadata.json already exists:
        if not os.path.isfile(path):
            initialize_metadata(path)

        #  add this model
        with open(os.path.join(file_path, "metadata.json"), "r+") as file:
            metadata_object = json.load(file)
            model_object = {
                "file_name": self.file_name,
                "area_under_curve": self.area_under_curve,
                "evaluation_plot_file_name": self.evaluation_plot_file_name
            }
            metadata_object['models'].append(model_object)

            # write to the top of the file and truncate the rest to overwrite contents in one "with" statement
            file.seek(0)
            json.dump(metadata_object, file)
            file.truncate()
