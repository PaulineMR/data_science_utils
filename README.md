# data_science_utils
Repository for my personal code about data science, data preprocessing and machine learning

## Prerequisites

- Python version 3.7 (or a pyenv environment with python 3.7)
- poetry


## Tests

To launch the tests use the following commands:
```sh
poetry install
poetry run nostests
```

## Content

### Preprocessing

#### Encoding

Implementation of the encoding methods:
- ordinal encoding
- one hot encoding
- target encoding

#### Feature extraction

Implementation of differents feature extraction methods:
- extract from a french city informations about its population and location
- extract from a feature a periodic version of itself
- extract from a date temporal informations and periodic temporal informations 

### Models

#### Multilabel classification

Implementation of the classifier chain and the ensemble of classifier chain used to resolve the problem of Multi-label Classification. Based on the papers: _Classifier Chains for Multi-label Classification (2009), Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank_ and _Assessing the multi-labelness of multi-label data (2019), Laurence A. F. Park, Yi Guo, Jesse Read_.

#### K-Means

Implementation of the K-means algorithm.

### Metric

#### Business metric

Implementation of a business metric

## What to improve?

- Ensemble of classifier chain should be multi-threaded
- Addition of new models (decision trees, random forest...)
- From periodic variable (cos and sin) come back to the original variable
- From a date and a city give the french holidays
- Implementation of the MAPE
- Give more information about the business metric than just the quantile
