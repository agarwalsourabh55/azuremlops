import argparse
from azureml.core import Dataset, Datastore
import utils


@utils.exec_time
def main(model_name):
    print("Connecting to Workspace and Data Store")
    ## Step 1- Connect to Workspace and Dataset
    ws = utils.retrieve_workspace()
    config = utils.get_model_config_ws(ws,model_name)

    datastore_name = config['datastore_name']
    datastore = Datastore.get(ws, datastore_name)  
    
    """
    Data preparation Script
    .
    .
    .
    .
    .
    .
    """
def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    args_parsed = parser.parse_args(args_list)
    return args_parsed

if __name__ == '__main__':
    args = parse_args()
    main(
        model_name=args.model_name
    )
