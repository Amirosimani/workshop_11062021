
import os
import json
import joblib
import numpy as np
import pandas as pd
from io import StringIO

from sagemaker_containers.beta.framework import worker


feature_columns_names = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                         'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                         'TAX', 'PTRATIO', 'B', 'LSTAT']
label_column = 'target'


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))
        print(df.head())

        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df = df.drop(columns=[label_column])
        
        print(f"Test data is loaded... {df[feature_columns_names].shape}")
        return df[feature_columns_names]
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def df(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


# def predict_fn(input_data, model):
#     """Preprocess input data

#     We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
#     so we want to use .transform().

#     The output is returned in the following order:

#         rest of features either one hot encoded or standardized
#     """
#     features = model.transform(input_data)

#     if label_column in input_data:
#         # Return the label (as the first column) and the set of features.
#         return np.insert(features, 0, input_data[label_column], axis=1)
#     else:
#         # Return only the set of features
#         return features


def model_fn(model_dir):
    """Deserialize fitted model
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
