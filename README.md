fmri_twins_prediction
==============================

Hello. We are Roland and Sven and this is our MLOps project!<br>

# Project Description

In our Project we intend to work with public data from OpenNeuro (https://openneuro.org/datasets/ds004169)
consisting of MRI scans of 1202 participants. Especially all participants are twins. We therefore want to
predict twin status based on functional/structural brain data. The data consists of MRI scans for each subject
yielding a functional and an anatomical scan in ‘.nii’ format which can be conveniently handled using nilearn
package (https://nilearn.github.io/stable/index.html). The same package could be used in a preprocessing
step for brain parcellation. In the prediction task we intend to work with pytorch geometric since functional
as well as anatomical connections can be interpreted as a graph (each brain region corresponds to a node and
functional/structural connection to an edge).
Ideally we have a preprocessing workflow followed by an prediction workflow and maybe an explanation step
which might show which brain regions/connections led to the decision of being a twin pair.


# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## :wave: Attribution
Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience
