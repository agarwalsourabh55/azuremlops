import argparse
import pandas as pd
from azureml.core import Dataset, Datastore, Run
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import utils


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


@utils.exec_time
def main(model_name, output_dir):
    # connect to ws
    print("Connecting to Workspace and Data Store")
    # Step 1- Connect to Workspace and Dataset
    ws = utils.retrieve_workspace()
    run = Run.get_context()
    config = utils.get_model_config_ws(ws, model_name)
    datastore_name = config['datastore_name']
    datastore = Datastore.get(ws, datastore_name)
    
    diabetes_module = Dataset.get_by_name(ws, name=config["data_prep_datasets"]["dataset"]+"_processed")
    diabetes = diabetes_module.to_pandas_dataframe()
    
    # Separate features and labels
    X, y = diabetes[['Pregnancies',
                     'PlasmaGlucose',
                     'DiastolicBloodPressure',
                     'TricepsThickness',
                     'SerumInsulin',
                     'BMI',
                     'DiabetesPedigree',
                     'Age']].values, diabetes['Diabetic'].values

    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Train adecision tree model
    print('Training a decision tree model...')
    model = DecisionTreeClassifier().fit(X_train, y_train)

    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    run.log('Accuracy', np.float(acc))

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    run.log('AUC', np.float(auc))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    run.log_image(name = "ROC", plot = fig)
    plt.show()

    # Save the trained model in the outputs folder
    print("Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, model_name+'.pkl')
    joblib.dump(value=model, filename=model_file)
    
    run.complete()
    
    
def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--model_name', type=str, default='deductions')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == '__main__':
    args = parse_args()
    main(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
