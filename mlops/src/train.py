import argparse
# BASE MODEL SAR
import pandas as pd
from azureml.core import Dataset, Datastore, Run
from recommenders.models.sar import SAR

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


@utils.exec_time
def main(model_name, output_dir):
    # connect to ws
    print("Connecting to Workspace and Data Store")
    # Step 1- Connect to Workspace and Dataset
    ws = utils.retrieve_workspace()
    config = utils.get_model_config_ws(ws, model_name)
    datastore_name = config['datastore_name']
    datastore = Datastore.get(ws, datastore_name)
    """
    Model Training Script
    .
    .
    .
    .
    .
    .
    
    """
  

def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastore_name', type=str)
    parser.add_argument('--dataset_name', type=str,
                        default='<your-dataset-name>')
    parser.add_argument('--model_name', type=str, default='<your-model-name>')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--config_path', type=str)
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == '__main__':
    args = parse_args()
    main(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
