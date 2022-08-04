import argparse
from azureml.core import Dataset, Datastore
import utils
from sklearn.preprocessing import MinMaxScaler
import os
from azureml.core import Run


@utils.exec_time
def main(model_name):
    print("Connecting to Workspace and Data Store")
    ## Step 1- Connect to Workspace and Dataset
    ws = utils.retrieve_workspace()
    run = Run.get_context()
    config = utils.get_model_config_ws(ws,model_name)
    
    datastore_name = config['datastore_name']
    datastore = Datastore.get(ws, datastore_name)  
    
    #load dataset
    diabetes_module = Dataset.get_by_name(ws, name=config["data_prep_datasets"]["dataset"])
    diabetes = diabetes_module.to_pandas_dataframe()
    
    # Log row count
    row_count = (len(diabetes))
    run.log('raw_rows', row_count)

    # remove nulls
    diabetes = diabetes.dropna()

    # Normalize the numeric columns
    scaler = MinMaxScaler()
    num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
    diabetes[num_cols] = scaler.fit_transform(diabetes[num_cols])

    # Log processed rows
    row_count = (len(diabetes))
    run.log('processed_rows', row_count)

    Dataset.Tabular.register_pandas_dataframe(diabetes,target = datastore, name= config["data_prep_datasets"]["dataset"]+"_processed")

#     # Save the prepped data
#     print("Saving Data...")
#     os.makedirs(dest_name, exist_ok=True)
#     save_path = os.path.join(dest_name,'data.csv')
#     diabetes.to_csv(save_path, index=False, header=True)

    # End the run
    run.complete()
def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
#     parser.add_argument('--pipeline_data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
    args_parsed = parser.parse_args(args_list)
    return args_parsed

if __name__ == '__main__':
    args = parse_args()
    main(
        model_name=args.model_name,
#         dest_name = args.pipeline_data
    )
