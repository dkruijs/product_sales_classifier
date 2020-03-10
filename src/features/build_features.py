# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path

import pandas as pd

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs feature engineering scripts to turn raw data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    historical = pd.read_csv(input_filepath)
    historical['added_feature'] = "w"

    historical.to_csv(output_filepath, index=False)


if __name__ == "__main__": 
    main()