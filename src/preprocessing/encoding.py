#####################
# Exposed functions #
#####################


def ordinal_encoding(df, categorical_variable):
    """
    Encode a categorical variable using the ordinal encoding method. This is useful when using an algorithm that only
    use numerical variable.
    The limitation of this method is that it creates an order in the variable that represent nothing (for example
    Paris=1 < Pau=2 < Nantes=3)
    The advantage of this method is that if the order has a meaning, then it gives better informations (for example
    small=1 < medium=2 < big=3).

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the categorical_variable
    categorical_variable: str
        name of the categorical variable to encode

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new encoded variable remplacing the original categorical_variable

    Examples:
    ---------
    >>> print(df)
         size
    0   small
    1  medium
    2   small
    3   small
    4     big
    5  medium
    >>> print(ordinal_encoding(df, "size")
      size
    0    0
    1    1
    2    0
    3    0
    4    2
    5    1
    """
    list_values_variable = df[categorical_variable].unique()
    for index, value_variable in enumerate(list_values_variable):
        df.loc[df[categorical_variable] == value_variable, :] = index
    return df


def ordinal_encoding_multiple_features(df, categorical_variables):
    """
    Encode multiple categorical variable using the ordinal encoding method.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the categorical_variables
    categorical_variables: list(str)
        list of the names of the categorical variable to encode

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new encoded variables remplacing the original categorical_variables
    """
    for categorical_variable in categorical_variables:
        df = ordinal_encoding(df, categorical_variable)
    return df


def one_hot_encoding(df, categorical_variable, prefix_new_feature="is_"):
    """
    Encode a categorical variable using the one hot encoding method. This is useful when using an algorithm that only
    use numerical variable.
    The advantage of this method is that the distance between points that have the same values for all the variables
    exept the categorical one are at equal distance. Which means that all the values of this particula variable are
    equivalents.
    The limitation of this method is that it adds a lot of dimension, which can be problematic when we don't have
    enough sample.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the categorical_variable
    categorical_variable: str
        name of the categorical variable to encode
    prefix_new_feature: str, optional
        prefix for the new variables, default is "is_"

    Returns
    ----------
    pd.DataFrame
        original DataFrame without the original categorical_variable and with the new encoded variables

    Examples:
    ---------
    >>> print(df)
        species
    0         0
    1         1
    2         2
    3         0
    >>> print(one_hot_encoding(df, "species")
            is_0   is_1   is_2
    0       True  False  False
    1      False   True  False
    2      False  False   True
    3       True  False  False
    """
    list_values_variable = df[categorical_variable].unique()
    for value_variable in list_values_variable:
        new_variable_name = prefix_new_feature + str(value_variable)
        df.loc[:, new_variable_name] = df[categorical_variable] == value_variable

    return df.drop(columns=categorical_variable)


def one_hot_encoding_multiple_features(
    df, categorical_variables, prefix_new_feature="is_"
):
    """
    Encode multiple categorical variable using the one hot encoding method.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the categorical_variable
    categorical_variables: list(str)
        list of the names of the categorical variable to encode
    prefix_new_feature: str, optional
        prefix for the new variables, default is "is_"

    Returns
    ----------
    pd.DataFrame
        original DataFrame without the original categorical_variables and with the new encoded variables
    """
    for categorical_variable in categorical_variables:
        df = one_hot_encoding(df, categorical_variable, prefix_new_feature)
    return df


def target_encoding(df, target_variable, categorical_variable):
    """
    Encode a categorical variable using the one hot encoding method. This is useful when using an algorithm that only
    use numerical variable and a numerical target.
    The advantage of this method is that it creates a new feature that explain the target.
    The limitation of this method is that it can create a lot of overfitting. Basically the distribution of the target
    variable for a particualr value of the categorical variable in the the training dataset can be different from the
    one in the test dataset.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the categorical_variable
    target_variable: str
        name of the target variable
    categorical_variable: str
        name of the categorical variable to encode

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new encoded variable remplacing the original categorical_variable

    Examples:
    ---------
    >>> print(df)
        size (cm)   animal
    0    30            cat
    1     9        hamster
    2    25            cat
    3    24            cat
    4    46            dog
    5    10        hamster
    >>> print(target_encoding(df, "target", "animal")
        size (cm)   animal
    0    30      26.333333
    1     9       9.500000
    2    25      26.333333
    3    24      26.333333
    4    46      46.000000
    5    10       9.500000
    """
    means = df.groupby(categorical_variable)[target_variable].mean()
    df[categorical_variable] = df[categorical_variable].map(means)

    return df


def target_encoding_multiple_features(df, target_variable, categorical_variables):
    """
    Encode multiple categorical variable using the target encoding method.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the categorical_variables
    target_variable: str
        name of the target variable
    categorical_variables: list(str)
        list of the names of the categorical variable to encode

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new encoded variables remplacing the original categorical_variables
    """
    for categorical_variable in categorical_variables:
        df = target_encoding(df, target_variable, categorical_variable)
    return df
