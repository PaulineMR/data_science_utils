"""
Name:           multilabel_classification
Author:         Pauline Martin
Description:    Models that are used to resolve the problem of Multi-label Classification. Based on the papers :
    Classifier Chains for Multi-label Classification (2009), Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank
    Assessing the multi-labelness of multi-label data (2019), Laurence A. F. Park, Yi Guo, Jesse Read
"""
# Standard imports
import copy
from random import shuffle

# External imports
from sklearn.tree import DecisionTreeClassifier

# Internal imports
from src.models.exceptions import NotFittedError

#####################
# Private functions #
#####################


def _list_to_dataframe(row, variable):
    """
    Tool function for series_of_list_to_dataframe
    """
    list_labels = row[variable]
    for label in list_labels:
        row[label] = 1
    return row


#####################
# Exposed functions #
#####################


def series_of_list_to_dataframe(df, variable, possible_values):
    """
    From a dataframe that contains a series of list, generate new columns that are binary that indicate if the
    value was present in the list.

    Parameters
    ----------
    df: pd.DataFrame
        training set that have at least the variable {variable} which is a list that contains
        value from {possible_values}
    variable: str
        name of the variable which contains lists
    possible_values: list(str)
        list of the possible values of the variable

    Returns
    ----------
    pd.DataFrame
        original dataframe without {variable} and with new columns with the name of the {possible_values}
    """
    df_temp = df.copy()
    for possible_value in possible_values:
        df_temp[possible_value] = 0
    return df_temp.apply(func=_list_to_dataframe, axis=1, variable=variable).drop(
        columns=variable
    )


def unique_value_series_of_list(series):
    """
    From a pd.Series of list get all the possible values that can be in those lists.

    Parameters
    ----------
    df: pd.Series(list(str))
        series to get all he possible values from

    Returns
    ----------
    list(str)
        possible values that can be in the lists in the series
    """
    return list(set(series.sum()))


###################
# Exposed classes #
###################


class ClassifierChainModel:
    """
    Implementation of the machine learning algorithm Classifier Chain.

    Attributes
    ----------
    labels: list(str)
        the output take as value a list of these labels, the order of the list will give the order of the model in
        the chain
    model: custom class
        class that represent a machine learning model for classification that is not fitted, should have the functions
        .fit(X, y) and .predict(X) (could be a sklearn models)

    Methods
    -------
    get_basic_model()
        Get the model that is used to create a chain.
    get_labels()
        Get the labels that the ClassifierChainModel is predicting.
    trained()
        Is the ClassifierChainModel trained?
    fit(df)
        Build and train a classifier chain from the training set.
    predict(df)
        Predict class value for the dataframe using the model.
    """

    #################
    # Init function #
    #################

    def __init__(
        self, labels, model=DecisionTreeClassifier(),
    ):
        """
        Init function for ClassifierChainModel.

        Parameters
        ----------
        labels: list(str)
            the output take as value a list of these labels
        model: custom class, optional
            class that represent a machine learning model for classification that is not fitted, should
            have the functions .fit(X, y) and .predict(X) (could be a sklearn models), default is
            sklearn.tree.DecisionTreeClassifier
        """
        self.labels = labels
        self.model = model

        self._chain_of_models = {}
        self._is_trained = False

    #####################
    # Exposed functions #
    #####################

    def get_basic_model(self):
        """
        Get the model that is used to create a chain.
        """
        return self.model

    def get_labels(self):
        """
        Get the labels that the ClassifierChainModel is predicting.
        """
        return self.labels

    def trained(self):
        """
        Is the ClassifierChainModel trained?
        """
        return self._is_trained

    def fit(self, df):
        """
        Build and train a classifier chain from the training set.

        Parameters
        ----------
        df: pd.DataFrame
            training set that have at least the variables {self.labels}
            and one other variable
        """
        # Separate the data from the output
        outputs = df[self.get_labels()]
        evolving_df = df.drop(columns=self.get_labels())

        chain_of_models = {}
        # For each label, in order, fit a model that will predict a target that is fed from the data and the target of
        # the previous model.
        for label in self.get_labels():
            # Get the output of the current model which is: Is this current label True?
            current_output = outputs[label]
            # Create a model and fit it to the current data
            current_model = copy.deepcopy(self.get_basic_model())
            current_model.fit(evolving_df, current_output)
            chain_of_models[label] = current_model

            # The new data include the truth about: Is the current label True?
            evolving_df[label] = current_output

        # Keep the models and certify that the model is trained
        self._chain_of_models = chain_of_models
        self._is_trained = True

    def predict(self, df):
        """
        Predict class value for the dataframe using the model.

        Parameters
        ----------
        df: pd.DataFrame
            data that has as columns the features used for the training of this model

        Returns
        ----------
        pd.DataFrame
            predicted classes
        """
        # Check that the model is trained
        if not self.trained():
            raise NotFittedError(
                "This ClassifierChainModel instance is not fitted yet. Call 'fit' with appropriate arguments before\
                using self estimator."
            )

        evolving_df = df.copy()
        # For each model in the chain, do the prediction and use the result as input for the next model
        for label, model in self._chain_of_models.items():
            evolving_df[label] = model.predict(evolving_df)

        # Compile the results of the models to obtain the results
        return evolving_df[self.get_labels()]


class EnsembleClassifierChainModel:
    """
    Public class
    Implementation of the machine learning algorithm Ensemble Classifier Chain.

    Attributes
    ----------
    labels: list(str)
        list of the possible values of the output
    model: custom class
        class that represent a machine learning model for classification that is not fitted, should have the functions
        .fit(X, y) and .predict(X) (could be a sklearn models)
    nb_estimator: int
        number of Classifier Chain in the ensemble
    subsample_bagging: float [0,1]
        proportion of the trained dataset that will be used to create one Classifier Chain

    Methods
    -------
    get_basic_model()
        Get the model that is used to create a chain.
    get_labels()
        Get the labels that the EnsembleClassifierChainModel is predicting.
    get_nb_estimator()
        Get the number of estimator in the EnsembleClassifierChainModel.
    trained()
        Is the EnsembleClassifierChainModel trained?
    fit(df)
        Build and train an ensemble classifier chain from the training set.
    predict(df)
        Predict class value for the dataframe using the model.
    """

    #################
    # Init function #
    #################

    def __init__(
        self,
        labels,
        model=DecisionTreeClassifier(),
        nb_estimator=100,
        subsample_bagging=0.1,
    ):
        """
        Init function for EnsembleClassifierChainModel.

        Parameters
        ----------
        labels: list(str)
            list of the possible values of the output
        model: custom class, optional
            class that represent a machine learning model for classification that is not fitted, should
            have the functions .fit(X, y) and .predict(X) (could be a sklearn models), default is
            sklearn.tree.DecisionTreeClassifier
        nb_estimator: int
            number of Classifier Chain in the ensemble, default is 100
        subsample_bagging: float [0,1]
            proportion of the trained dataset that will be used to create one Classifier Chain, default is 0.1
        """
        self.labels = labels
        self.model = model
        self.nb_estimator = nb_estimator
        self._subsample_bagging = subsample_bagging

        self._ensemble_of_models = []
        self._is_trained = False

    #####################
    # Private functions #
    #####################

    def _get_subsample_data(self, df):
        """
        Get a subsample of df to make the prediction.

        Parameters
        ----------
        df: pd.DataFrame
            training set
        """
        return df.sample(frac=self._subsample_bagging, replace=False, axis=0)

    def _voting_for_final_labels(self, predictions):
        """
        Make the classifier chains vote for the samples labels using their predictions on these sample.

        Parameters
        ----------
        predictions: list(pd.DataFrame)
            list of the predictions given by the classifier chains

        Returns
        ----------
            pd.DataFrame
                final prediction after the vote
        """
        # Sum the labels columns of each prediction
        prediction_vote = predictions[0][self.get_labels()]
        for i in range(1, len(predictions)):
            for label in prediction_vote.columns:
                prediction_vote.loc[:, label] = prediction_vote.loc[:, label].add(
                    predictions[i].loc[:, label]
                )
        # A label is predicted for this sample if at least 3/4 of the models said that this is correct
        prediction_vote[prediction_vote < self.get_nb_estimator() * 3 / 4] = 0
        prediction_vote[prediction_vote != 0] = 1

        return prediction_vote

    #####################
    # Exposed functions #
    #####################

    def get_basic_model(self):
        """
        Get the model that is used to create a chain.
        """
        return self.model

    def get_labels(self):
        """
        Get the labels that the EnsembleClassifierChainModel is predicting.
        """
        return self.labels

    def get_nb_estimator(self):
        """
        Get the number of estimator in the EnsembleClassifierChainModel.
        """
        return self.nb_estimator

    def trained(self):
        """
        Is the EnsembleClassifierChainModel trained?
        """
        return self._is_trained

    def fit(self, df):
        """
        Build and train an ensemble classifier chain from the training set.

        Parameters
        ----------
        df: pd.DataFrame
            training set that have at least the variable {self.output_variable}
            and one other variable
        """
        ensemble_of_models = []
        # Create nb_estimator models
        for _ in range(self.get_nb_estimator()):
            # Get a subsample of the dataset
            current_dataset = self._get_subsample_data(df)
            # Randomize the order of the labels
            current_labels = copy.deepcopy(self.get_labels())
            shuffle(current_labels)
            # Create the ClassifierChainModel with the labels in a random order and fit it
            current_model = ClassifierChainModel(current_labels, self.get_basic_model())
            current_model.fit(current_dataset)
            ensemble_of_models.append(current_model)
        self._ensemble_of_models = ensemble_of_models
        self._is_trained = True

    def predict(self, df):
        """
        Predict class value for the dataframe using the model.

        Parameters
        ----------
        df: pd.DataFrame
            data that has as columns the features used for the training of this model

        Returns
        ----------
        pd.Series
            predicted classes
        """
        # Check that the model is trained
        if not self.trained():
            raise NotFittedError(
                "This EnsembleClassifierChainModel instance is not fitted yet. Call 'fit' with appropriate arguments before\
                using this estimator."
            )

        all_predictions = []
        # For each nb_estimator models in the ensemble do the prediction
        for model in self._ensemble_of_models:
            all_predictions.append(model.predict(df))

        # The models vote for the labels
        return self._voting_for_final_labels(all_predictions)
