import argparse
from azureml.core import Run, Model
from azureml.exceptions import WebserviceException


def main(model_name):

    run = Run.get_context()  # Will fail if offline run, evaluation is only supported in AML runs
    ws = run.experiment.workspace

    try:
        # Retrieve latest model registered with same model_name
        model_registered = Model(ws, model_name)

        if is_new_model_better(run, model_registered):
            print("New trained model is better than existing model.")
        else:
            print("New trained model is not better than latest model. Canceling job.")
            run.parent.cancel()

    except WebserviceException:
        print("First model.")

    print('Model should be registered. Proceeding to next step.')


def is_new_model_better(run, old_model):
    metrics_new_model = run.get_metrics()
    metrics_old_model = old_model.tags
    # Compare Models Here
    # TODO
    return True


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='The input from previous steps')
    parser.add_argument('--model_name', type=str, help='The name of the model file')
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    main(args.model_name)
