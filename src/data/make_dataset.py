# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
from sklearn import preprocessing


def preprocess_data(data, labelencoder_dict):
    data.loc[:,'MarketingTypeD'] = labelencoder_dict['MarketingTypeD'].transform(data.MarketingType)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = pd.read_csv(input_filepath)

    historical = data[data.File_Type == 'Historical']
    
    label_encoder = {'MarketingTypeD': preprocessing.LabelEncoder().fit(historical.MarketingType)}
    preprocess_data(historical, label_encoder)
    
    # active = data[data.File_Type == 'Active']

    historical.to_csv(output_filepath, index=False)
    # active.to_csv(output_filepath)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    # import os
    # main(os.path.join(project_dir, "data", "raw", "SalesKaggle3.csv"), os.path.join(project_dir, "data", "processed", "historical.csv"))

    main()
