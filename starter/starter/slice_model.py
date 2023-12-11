# Calculating performances on model slices.
from sklearn.model_selection import train_test_split
import logging 
from ml.model import compute_model_metrics, inference
from ml.data import process_data
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add code to load in the data.
logging.info("loading the cleaned census data")
df_input = pd.read_csv("/Users/seangao/Desktop/Deploy-ML-model/starter/data/census_cleaned.csv", header = 0)
logging.info("defining categorical features")
# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def calc_model_slices(df, feature_name):
    '''
    Obtain model slices based on the feature name (preferably cat feature) as 
    user selected

    Input:
        df: original datafrmae
        feature_name: name of the feature to compute model slice
    Output:
        list to be Exported to slice_output.txt with each line folloiwng the format of
        (feature_name, feature_value, n samples, precision, recall, fbeta)
    '''
    rf_model_path = "/Users/seangao/Desktop/Deploy-ML-model/starter/model/rf_model.pkl"
    encoder_path = "/Users/seangao/Desktop/Deploy-ML-model/starter/model/encoder.pkl"
    lb_path = "/Users/seangao/Desktop/Deploy-ML-model/starter/model/labelbinarizer.pkl"
    logging.info('loading ml model artifacts')
    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    logging.info('loading encoder model artifacts')
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    logging.info('loading lb model artifacts')
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)

    logging.info('conducting train test split')
    train, test = train_test_split(df, test_size=0.20)
    output = []
    for val in test[feature_name].unique():
        df_cur = test[test[feature_name] == val].reset_index(drop = True)
        X_test, y_test, _, _ = process_data(
            df_cur,
            cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            training=False)
        
        y_pred = inference(rf_model,X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        cur_output = f"{feature_name} = {val}:: Precision: {precision}. Recall: {recall}. Fbeta: {fbeta}"
        output.append(cur_output)
    return output

if __name__ == '__main__':
    final_output = []
    for feature in cat_features:
        final_output += calc_model_slices(df_input, feature)
    logging.info('writing to output')
    with open('/Users/seangao/Desktop/Deploy-ML-model/starter/model/slice_output.txt', 'w') as file:
        for content in final_output:
            file.write(content + '\n')