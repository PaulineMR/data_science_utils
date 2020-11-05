"""
Name:           feature_extraction
Author:         Pauline Martin
Description:    Methods of feature extraction. Currently this repositor can extract features from:
                    - city
                    - periodic feature
                    - date
"""

# Standard imports
import os
import numpy as np

# External imports
import pandas as pd

# Global variables
CITY_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data")
CITY_DATASET = "insee_city.csv"


#####################
# Private functions #
#####################


def _simplify_string_variable(series_string):
    # Remove the accents
    series_string = (
        series_string.str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    # Remove the capital letters
    series_string = series_string.str.lower()
    # Remove the dashes
    series_string = series_string.str.replace("-", "")
    # Remove the spaces
    series_string = series_string.str.replace(" ", "")

    return series_string


#####################
# Exposed functions #
#####################


def city_encoding(df, city_variable):
    """
    Encode a variable that represent a french city. This is useful when we want to extract
    informations from the city.
    The new features will be CODE_POSTAL (postal code), SUPERFICIE (surface), POPULATION, NB_MEN_FISC
    (number of fiscalised people), MED_REVENUE (median revenue), LATITUDE, LONGITUDE, CODE_DEPT
    (department code), NOM_DEPT (name of the department), NOM_REG (name of the region).
    WARNING: the naming of the cities can be problematic when using this function. for example
    the cities that start with "Saint" should be completely written (instead of using "St" or "St.).
    Moreover some city name exists to designate two different cities in two different department,
    in that case the function only get the city with the departement with a smaller number.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the city_variable
    city_variable: str
        name of the variable that represent the cities

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new added variables

    Examples:
    ---------
    >>> print(df)
                ville
    0          Paris
    1           Lyon
    2  Saint-Étienne
    3       Vauxbuin
    >>> print(city_encoding(df, "ville"))
                ville INSEE_COM  CODE_POSTAL  SUPERFICIE  ... NOM_DEPT                      NOM_REG
    0          Paris     75101        75001         182   ...    PARIS                ILE-DE-FRANCE
    1           Lyon     69381        69001         153   ...    RHONE         AUVERGNE-RHONE-ALPES
    2  Saint-Étienne     42218        42100        7996   ...    LOIRE         AUVERGNE-RHONE-ALPES
    3       Vauxbuin     02770         2200         506   ...    AISNE  NORD-PAS-DE-CALAIS-PICARDIE
    """
    df_copy = df.copy()

    # Simplify the name of the cities
    df_copy["basic_city_name"] = df_copy[city_variable].copy()
    df_copy["basic_city_name"] = _simplify_string_variable(df_copy["basic_city_name"])

    # Load the dataset with the informations about the cities
    df_cities = pd.read_csv(os.path.join(CITY_DATASET_PATH, CITY_DATASET)).drop(
        columns=["NOM_COM_2", "NOM_COM_3"]
    )
    # In case of big cities remove the arrondissement information
    df_cities.loc[
        df_cities["NOM_COM"] == "PARIS-1ER-ARRONDISSEMENT", "NOM_COM"
    ] = "PARIS"
    df_cities.loc[
        df_cities["NOM_COM"] == "MARSEILLE-1ER-ARRONDISSEMENT", "NOM_COM"
    ] = "MARSEILLE"
    df_cities.loc[df_cities["NOM_COM"] == "LYON-1ER-ARRONDISSEMENT", "NOM_COM"] = "LYON"
    # Simplify the name of the cities
    df_cities["NOM_COM"] = _simplify_string_variable(df_cities["NOM_COM"])
    df_cities.drop_duplicates(subset="NOM_COM", keep="first", inplace=True)
    # Make the departement and the postal name into string
    df_cities["CODE_DEPT"] = df_cities["CODE_DEPT"].astype(str)
    df_cities["CODE_DEPT"] = df_cities["CODE_DEPT"].apply(lambda x: "{0:0>2}".format(x))
    df_cities["CODE_POSTAL"] = df_cities["CODE_POSTAL"].astype(str)
    df_cities["CODE_POSTAL"] = df_cities["CODE_POSTAL"].apply(
        lambda x: "{0:0>5}".format(x)
    )

    # Merge the two datasets
    df_copy = df_copy.merge(df_cities, left_on="basic_city_name", right_on="NOM_COM")
    return df_copy.drop(columns=["basic_city_name"])


def generate_periodic_variables(
    df, periodic_variable, modulo_variable, drop_original_variable=True
):
    """
    Get from a periodic variable new informations that can be used to create a model with a machine learning
    algorithm. As those informations are periodics (for example after angle=359 it is angle=0) it
    is interesting to add the information about periodicity.
    The new features will be cos_{periodic_variable}, sin_{periodic_variable}.
    Blogpost on the subject: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the periodic_variable
    periodic_variable: str
        name of the variable that is periodic
    modulo_variable: int
        the periodic_variable has a period of modulo_variable, that is to say that in the periodic_variable
        0 is equivalent to modulo_variable and that periodic_variable is between 0 and modulo_variable-1
    drop_original_variable: bool, optional
        if True the periodic_variable is dropped from the dataframe. Default is True

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new added variables

    def _get_final_label(this, df, feature_name):
        df.loc[:, feature_name] = None
        for label in this.labels:
            df.loc[df[label], feature_name] = label
        return df.drop(columns=this.labels)

    Examples:
    ---------
    >>> print(df)
    >>> print(def generate_periodic_variables(df, "angle", 360, True)
    """
    df_copy = df.copy()

    # Check the values of the periodic variable
    if df_copy[periodic_variable].max() > modulo_variable:
        raise ValueError(
            f"modulo_variable ({modulo_variable}) is smaller than the maximum value of the periodic_variable \
            ({df_copy[periodic_variable].max()})"
        )

    # Place the variable on a trigonometric circle
    df_copy[periodic_variable] = df_copy[periodic_variable] * (
        2 * np.pi / modulo_variable
    )
    # Get the cosinus and the sinus
    df_copy[f"cos_{periodic_variable}"] = np.cos(df_copy[periodic_variable])
    df_copy[f"sin_{periodic_variable}"] = np.sin(df_copy[periodic_variable])

    if drop_original_variable:
        return df_copy.drop(columns=periodic_variable)
    return df_copy


def get_temporal_variables(df, date_variable):
    """
    Get from a date variable new informations that can be used to create a model with a machine learning
    algorithm.
    The new features will be time, day_of_week, day_of_month, month and year.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the date_variable which contains pd.Timestamp
    date_variable: str
        name of the variable that represent the date

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new added variables

    Examples:
    ---------
    >>> print(df)
                                 date
    0       2020-01-31 10:00:00+00:00
    1       2020-02-01 15:00:00+00:00
    2       2020-03-15 00:00:00+00:00
    >>> print(def get_temporal_variables(df, "date")
                                 date  time  day_of_week  day_of_month  month  year
    0       2020-01-31 10:00:00+00:00    10            4            31      1  2020
    1       2020-02-01 15:00:00+00:00    15            5             1      2  2020
    2       2020-03-15 00:00:00+00:00     0            6            15      3  2020
    """
    df_copy = df.copy()

    # Extract the informations about the date
    df_copy["time"] = df_copy[date_variable].dt.hour
    df_copy["day_of_week"] = df_copy[date_variable].dt.dayofweek
    df_copy["day_of_month"] = df_copy[date_variable].dt.day
    df_copy["month"] = df_copy[date_variable].dt.month
    df_copy["year"] = df_copy[date_variable].dt.year

    return df_copy


def get_periodic_temporal_variable(df, date_variable):
    """
    Get from a date variable new informations that can be used to create a model with a machine learning
    algorithm. As those informations are periodics (for example after time=23:59 it is time=00:00) it
    is interesting to add the information about periodicity.
    The new features will be cos_time, sin_time, cos_day, sin_day, cons_month, sin_month and year.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the date_variable
    date_variable: str
        name of the variable that represent the date

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new added variables

    Examples:
    ---------
    >>> print(df)
                                 date
    0       2020-01-31 10:00:00+00:00
    1       2020-02-01 15:00:00+00:00
    2       2020-03-15 00:00:00+00:00
    >>> print(def get_periodic_temporal_variable(df, "date")
                                 date  year  cos_time  sin_time  ...  cos_month  sin_month
    0       2020-01-31 10:00:00+00:00  2020 -0.866025  0.500000        1.000000   0.000000
    1       2020-02-01 15:00:00+00:00  2020 -0.707107 -0.707107        0.866025   0.500000
    2       2020-03-15 00:00:00+00:00  2020  1.000000  0.000000        0.500000   0.866025
    """
    df_copy = df.copy()

    # Get the temporal variables
    temporal_df = get_temporal_variables(df_copy, date_variable)

    # Transform them into periodic variables
    temporal_df = generate_periodic_variables(
        temporal_df,
        periodic_variable="time",
        modulo_variable=24,
        drop_original_variable=True,
    )
    temporal_df = generate_periodic_variables(
        temporal_df,
        periodic_variable="day_of_week",
        modulo_variable=7,
        drop_original_variable=True,
    )
    # Day of month should be between 0->30 (and not 1->31)
    temporal_df["day_of_month"] = temporal_df["day_of_month"] - 1
    temporal_df = generate_periodic_variables(
        temporal_df,
        periodic_variable="day_of_month",
        modulo_variable=31, # Decided to put on the max number of day
        drop_original_variable=True,
    )
    # Month should be between 0->11 (and not 1->12)
    temporal_df["month"] = temporal_df["month"] - 1
    temporal_df = generate_periodic_variables(
        temporal_df,
        periodic_variable="month",
        modulo_variable=12,
        drop_original_variable=True,
    )

    return temporal_df
