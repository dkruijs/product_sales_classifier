import os
import json


class Model(object):
    """The Model class collects a model binary with model metadata."""

    def __init__(self, file_name, model_object, area_under_curve, evaluation_plot_file):
        
        file_name = self.file_name
        model_object = self.model_object
        area_under_curve = self.area_under_curve
        #TODO:
        evaluation_plot_file = self.evaluation_plot_file

    def add_to_metadata(self, file_path):
        """Adds a model to the metadata.json file."""
        # metadata.json already exists: add this model
        if os.path.isfile(os.path.join(file_path, "models.json")):
            with open(os.path.join(file_path, "models.json"), "wb") as file:
                metadata_object = json.read(file)
                model_object = {
                    "file_name": self.file_name,
                    "area_under_curve": self.area_under_curve
                }
                metadata_object.models.append(model_object)
                json.dump(metadata_object, file)

        # metadata.json does not exist yet: create and add this model
        else:
            with open(os.path.join(file_path, "models.json"), "wb") as file:
                json_object = {
                    "models": [{
                        "file_name": self.file_name,
                        "area_under_curve": self.area_under_curve
                    }]
                }
                json.dump(json_object, file)
