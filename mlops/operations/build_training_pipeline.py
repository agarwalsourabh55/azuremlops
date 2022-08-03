import os
import argparse
import os
from azureml.core import Datastore, Environment
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

from utils import config, workspace, compute, pipeline

def main(model_name, datastore_name,train_script, eval_script, data_prep_script, pipeline_name, compute_name, environment_path, pipeline_version=None):

    # Connect to configured Workspace
    ws = workspace.retrieve_workspace()

    # Define Compute Target
    compute_target = compute.get_compute_target(ws, compute_name, config_file_path = f"{environment_path}/compute.yml")

    # Load Environemnt Configuration (Alternative is pre-existing ENV)
    env = Environment.load_from_directory(path=environment_path)

    ## Initialise Run configuration object
    run_config = RunConfiguration()
    run_config.environment = env

    datastore = Datastore.get(ws, datastore_name)
    # Create a PipelineData to pass data between steps
    pipeline_data = PipelineData('pipeline_data', datastore)

    # Create steps
    if data_prep_script:
        data_prep_step = PythonScriptStep(
            name=f"Dataprocessing for {model_name} Model",
            source_directory="mlops/src",
            script_name=data_prep_script,
            compute_target=compute_target,
            arguments=[
                '--model_name', model_name
            ],
            runconfig=run_config,
            allow_reuse=False
        )

    train_step = PythonScriptStep(
        name=f"Train {model_name} Model",
        source_directory="mlops/src",
        script_name=train_script,
        compute_target=compute_target,
        outputs = [pipeline_data],
        arguments=[
            '--model_name', model_name, 
            '--output_dir', pipeline_data
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    evaluate_step = PythonScriptStep(
        name=f"Evaluate {model_name} Model",
        source_directory="mlops/operations",
        script_name=eval_script,
        compute_target=compute_target,
        inputs=[pipeline_data],
        arguments=[
            '--model_dir', pipeline_data,
            '--model_name', model_name
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    register_step = PythonScriptStep(
        name=f"Register {model_name} Model",
        source_directory="mlops/operations",
        script_name="register_model.py",
        compute_target=compute_target,
        inputs=[pipeline_data],
        arguments=[
            '--model_dir', pipeline_data,
            '--model_name', model_name
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    # Set the sequence of steps in a pipeline
    if data_prep_script:
        train_step.run_after(data_prep_step)
        evaluate_step.run_after(train_step)
        register_step.run_after(evaluate_step)
        pipeline_steps = [data_prep_step, train_step, evaluate_step, register_step]
    else:
        evaluate_step.run_after(train_step)
        register_step.run_after(evaluate_step)
        pipeline_steps = [train_step, evaluate_step, register_step]

    # Publish training pipeline
    published_pipeline = pipeline.publish_pipeline(
        ws,
        name=pipeline_name,
        steps=pipeline_steps,
        description=f"Model training/retraining pipeline for {model_name}",
        version=pipeline_version
    )

    # Deploy Model Configuration
    workspace_datastore=Datastore.get(ws,  'workspaceblobstore')
    workspace_datastore.upload_files(files=[config_path], 
                                    target_path= "configuration",
                                    overwrite= True)

    print(f"Published pipeline {published_pipeline.name} version {published_pipeline.version}")


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--config_path", type=str)       
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    config_path = args.config_path
    config = config.get_model_config(config_path,model_name)
    main(
        datastore_name = config['datastore_name'],
        train_script = config['train_script'],
        eval_script = config['eval_script'],
        data_prep_script = config['train_data_prep_script'],
        model_name = config['model_name'],
        pipeline_name = config['train_pipeline_name'],
        compute_name = config['train_compute_name'],
        environment_path = config['environment_path'],
        pipeline_version = config['version']
    )
 