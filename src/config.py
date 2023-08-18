from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    SVC,
    SVR,
)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from skopt import BayesSearchCV

"""
Some boilerplate configs for the project
"""

TRAINING_DATA = "../data/data.csv"

LAGS = [10, 20]

PARAM_GRID_TREE = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [i for i in range(100, 2000, 100)],
    "min_samples_leaf": [i for i in range(100, 2000, 100)],
    "max_features": ["sqrt", "log2", None],
}

PARAM_GRID_SVM = {
    "C":"
}

GENETIC_PARAM_GRID_TREE = {
"max_depth": Integer(5, 6),
"min_samples_split": Integer(100, 200),
"max_features": Categorical(["log2"]),
}

"""
Models and CV's to use
"""

model_combos = {
    "Linear": (LinearSVC, LinearSVR),
    "Nu": (NuSVC, NuSVR),
    "SVC": (SVC, SVR),
}

cross_validation = {
    "GridSearchCV": GridSearchCV,
    "RandomizedSearchCV": RandomizedSearchCV,
    "HalvingGridSearchCV": HalvingGridSearchCV,
    "HalvingRandomSearchCV": HalvingRandomSearchCV,
    "BayesSearchCV": BayesSearchCV,
    "GASearchCV": GASearchCV,
}