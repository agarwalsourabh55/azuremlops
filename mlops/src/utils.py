import os
import sys
from time import time
import joblib
import pandas as pd
from azureml.core import Dataset, Model, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.run import _OfflineRun
from sklearn.metrics import *
from azureml.core import Datastore

def retrieve_workspace() -> Workspace:
    ws = None

    try:
        run = Run.get_context()
        if not isinstance(run, _OfflineRun):
            ws = run.experiment.workspace
            return ws
    except Exception as ex:
        print('Workspace from run not found', ex)

    try:
        ws = Workspace.from_config()
        return ws
    except Exception as ex:
        print('Workspace config not found in local folder', ex)

    try:
        sp = ServicePrincipalAuthentication(
            tenant_id=os.environ['AML_TENANT_ID'],
            service_principal_id=os.environ['AML_PRINCIPAL_ID'],
            service_principal_password=os.environ['AML_PRINCIPAL_PASS']
        )
        ws = Workspace.get(
            name="<ml-example>",
            auth=sp,
            subscription_id="<your-sub-id>"
        )
    except Exception as ex:
        print('Workspace config not found in project', ex)

    return ws

def exec_time(func):
    def run_time(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()

        print(f"Module: {func.__name__} exeution lasted: {(te-ts):.2f} seconds")
        return result

    return run_time

def get_dataset(ws, filename=None, path_datastore=None):
    """Get a dataset.

    Args:
        ws (Workspace): The Azure Machine Learning workspace object
        filename (str): The name of VM (compute target)
        path_datastore (str): The path to a model file (including file name)

    Returns:
        pandas DataFrame

    """
    df = None

    # get the data when run by external scripts
    try:
        run = Run.get_context()
        if not isinstance(run, _OfflineRun):
            dataset = run.input_datasets[filename]
            df = dataset.to_pandas_dataframe()
            print('Dataset retrieved from run')
            return df
    except Exception:
        print('Cannot retrieve dataset from run. Trying to get it from datastore by dataset name...')
    # get dataset from Dataset registry
    try:
        dataset = Dataset.get_by_name(ws, filename)
        df = dataset.to_pandas_dataframe()
        print('Dataset retrieved from datastore by dataset name')
        return df
    except Exception:
        print('Cannot retrieve dataset from datastore by dataset name. Trying to get it from datastore by path...')
    # get dataset directly from datastore
    try:
        datastore = ws.get_default_datastore()
        dataset = Dataset.Tabular.from_delimited_files(path=(datastore, path_datastore))
        df = dataset.to_pandas_dataframe()
        print('Dataset retrieved from datastore by path')
        return df
    except Exception:
        print('Cannot retrieve a dataset from datastore by path. Trying to get it from a local CSV file...')
    # get dataset from a local CSV file
    try:
        df = pd.read_csv(filename)
        print('Dataset retrieved from a local CSV file')
        return df
    except Exception:
        print('Cannot retrieve a dataset from a local CSV file.')

    if df is None:
        print('Cannot retrieve a dataset. Exiting.')
        sys.exit(-1)

    return df


def get_model(ws, model_name, model_version=None, model_path=None):
    """Get or create a compute target.

    Args:
        ws (Workspace): The Azure Machine Learning workspace object
        model_name (str): The name of ML model
        model_version (int): The version of ML model (If None, the function returns latest model)
        model_path (str): The path to a model file (including file name). Used to load a model from a local path.

    Returns:
        Model/model object: The trained model (if it is already registered in AML workspace,
                               then Model object is returned. Otherwise, a model object loaded with
                               joblib is returned)

    """
    model = None

    try:
        model = Model(ws, name=model_name, version=model_version)
        print(f'Found the model by name {model_name} and version {model_version}')
        return model
    except Exception:
        print((f'Cannot load a model from AML workspace by model name {model_name} and model_version {model_version}. '
               'Trying to load it by name only.'))
    try:
        models = Model.list(ws, name=model_name, latest=True)
        if len(models) == 1:
            print(f'Found the model by name {model_name}')
            model = models[0]
            return model
        elif len(models) > 1:
            print('Expected only one model.')
        else:
            print('Empty list of models.')
    except Exception:
        print((f'Cannot load a model from AML workspace by model name {model_name}. '
               'Trying to load it from a local path.'))

    try:
        model = joblib.load(model_path)
        print('Found the model by local path {}'.format(model_path))
        return model
    except Exception:
        print('Cannot load a model from {}'.format(model_path))

    if model is None:
        print('Cannot load a model. Exiting.')
        sys.exit(-1)

    return model

import json

def get_model_config(model_config_path,model_name):
    config = json.load(open(model_config_path, 'r'))
    model_config = config[model_name]
    return model_config

import os
def get_model_config_ws(ws, model_name, local=False):
    output_dir = "outputs"
    if not local:
        workspace_datastore=Datastore.get(ws,'workspaceblobstore')
    
        workspace_datastore.download(target_path = output_dir, 
                                prefix='configuration/model_config.json',
                                overwrite= True)    
    config = json.load(open(f"{output_dir}/configuration/model_config.json", 'r'))
    model_config = config[model_name]
    return model_config