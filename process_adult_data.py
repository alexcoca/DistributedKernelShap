import argparse
import pickle
import logging
import os
import requests

import numpy as np
import pandas as pd

from io import StringIO
from requests import RequestException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from typing import Any, Dict, List, Tuple, Union
from utils.utils import Bunch

logger = logging.getLogger(__name__)

ADULT_URLS = [
    'https://storage.googleapis.com/seldon-datasets/adult/adult.data',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data',
]  # type: List[str]


def fetch_adult(features_drop: list = None, return_X_y: bool = False, url_id: int = 0) -> \
        Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads and pre-processes 'adult' dataset.
    More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

    Parameters
    ----------
    features_drop
        List of features to be dropped from dataset, by default drops ["fnlwgt", "Education-Num"]
    return_X_y
        If true, return features X and labels y as numpy arrays, if False return a Bunch object
    url_id
        Index specifying which URL to use for downloading

    Returns
    -------
    Bunch
        Dataset, labels, a list of features and a dictionary containing a list with the potential categories
        for each categorical feature where the key refers to the feature column.
    (data, target)
        Tuple if ``return_X_y`` is true
    """
    if features_drop is None:
        features_drop = ["fnlwgt", "Education-Num"]

    # download data
    dataset_url = ADULT_URLS[url_id]
    raw_features = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital Status',
                    'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
                    'Hours per week', 'Country', 'Target']
    try:
        resp = requests.get(dataset_url)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise

    raw_data = pd.read_csv(StringIO(resp.text), names=raw_features, delimiter=', ', engine='python').fillna('?')

    # get labels, features and drop unnecessary features
    labels = (raw_data['Target'] == '>50K').astype(int).values
    features_drop += ['Target']
    data = raw_data.drop(features_drop, axis=1)
    features = list(data.columns)

    # map categorical features
    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates'
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }
    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    married_map = {
        'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
    }
    mapping = {'Education': education_map, 'Occupation': occupation_map, 'Country': country_map,
               'Marital Status': married_map}

    data_copy = data.copy()
    for f, f_map in mapping.items():
        data_tmp = data_copy[f].values
        for key, value in f_map.items():
            data_tmp[data_tmp == key] = value
        data[f] = data_tmp

    # get categorical features and apply labelencoding
    categorical_features = [f for f in features if data[f].dtype == 'O']
    category_map = {}
    for f in categorical_features:
        le = LabelEncoder()
        data_tmp = le.fit_transform(data[f].values)
        data[f] = data_tmp
        category_map[features.index(f)] = list(le.classes_)

    # only return data values
    data = data.values
    target_names = ['<=50K', '>50K']

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)


def load_adult_dataset():
    """
    Load the Adult dataset.
    """

    logging.info("Preprocessing data...")
    return fetch_adult()


def preprocess_adult_dataset(dataset, seed=0, n_train_examples=30000) -> Dict[str, Any]:
    """
    Splits dataset into train and test subsets and preprocesses it.
    """

    logging.info("Splitting data...")

    np.random.seed(seed)
    data = dataset.data
    target = dataset.target
    data_perm = np.random.permutation(np.c_[data, target])
    data = data_perm[:, :-1]
    target = data_perm[:, -1]

    X_train, y_train = data[:n_train_examples, :], target[:n_train_examples]
    X_test, y_test = data[n_train_examples + 1:, :], target[n_train_examples + 1:]

    logging.info("Transforming data...")
    category_map = dataset.category_map
    feature_names = dataset.feature_names

    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = StandardScaler()

    categorical_features = list(category_map.keys())
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='error')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # create groups for categorical variables
    numerical_feats_idx = preprocessor.transformers_[0][2]
    categorical_feats_idx = preprocessor.transformers_[1][2]
    ohe = preprocessor.transformers_[1][1]

    # compute encoded dimension; -1 as ohe is setup with drop='first'
    feat_enc_dim = [len(cat_enc) - 1 for cat_enc in ohe.categories_]
    num_feats_names = [feature_names[i] for i in numerical_feats_idx]
    cat_feats_names = [feature_names[i] for i in categorical_feats_idx]

    group_names = num_feats_names + cat_feats_names
    # each sublist contains the col. indices for each variable in group_names
    groups = []
    cat_var_idx = 0

    for name in group_names:
        if name in num_feats_names:
            groups.append(list(range(len(groups), len(groups) + 1)))
        else:
            start_idx = groups[-1][-1] + 1 if groups else 0
            groups.append(list(range(start_idx, start_idx + feat_enc_dim[cat_var_idx])))
            cat_var_idx += 1

    return {
        'X': {
            'raw': {'train': X_train, 'test': X_test},
            'processed': {'train': X_train_proc, 'test': X_test_proc}},
        'y': {'train': y_train, 'test': y_test},
        'preprocessor': preprocessor,
        'orig_feature_names': feature_names,
        'groups': groups,
        'group_names': group_names,
    }


def main():

    if not os.path.exists('data'):
        os.mkdir('data')

    # load and preprocess data
    adult_dataset = load_adult_dataset()
    adult_preprocessed = preprocess_adult_dataset(adult_dataset, n_train_examples=args.n_train_examples)
    # select first args.n_background_samples in train set as background dataset
    background_dataset = {'X': {'raw': None, 'preprocessed': None}, 'y': None}
    n_examples = args.n_background_samples
    background_dataset['X']['raw'] = adult_preprocessed['X']['raw']['train'][0:n_examples, :]
    background_dataset['X']['preprocessed'] = adult_preprocessed['X']['processed']['train'][0:n_examples, :]
    background_dataset['y'] = adult_preprocessed['y']['train'][0:n_examples]
    with open('data/adult_background.pkl', 'wb') as f:
        pickle.dump(background_dataset, f)
    with open('data/adult_processed.pkl', 'wb') as f:
        pickle.dump(adult_preprocessed, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_background_samples', type=int, default=100, help="Background set size.")
    parser.add_argument('-n_train_examples', type=int, default=30000, help="Number of training examples.")
    args = parser.parse_args()
    main()
