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
    # Simplify the name of the cities
    df["basic_city_name"] = df[city_variable].copy()
    df["basic_city_name"] = _simplify_string_variable(df["basic_city_name"])

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

    # Merge the two datasets
    df = df.merge(df_cities, left_on="basic_city_name", right_on="NOM_COM")
    return df.drop(columns=["basic_city_name", "NOM_COM"])


def generate_periodic_variables(
    df, periodic_variable, modulo_variable, drop_original_variable=True
):
    """
    Get from a periodic variable new informations that can be used to create a model with a machine learning
    algorithm. As those informations are periodics (for example after angle=359 it is angle=0) it
    is interesting to add the information about periodicity.
    The new features will be cos_{periodic_variable}, sin_{periodic_variable}.

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
    if df[periodic_variable].max() > modulo_variable:
        raise ValueError(
            f"modulo_variable ({modulo_variable}) is smaller than the maximum value of the periodic_variable \
            ({df[periodic_variable].max()})"
        )

    # Place the variable on a trigonometric circle
    df[periodic_variable] = df[periodic_variable] * (2 * np.pi / modulo_variable)
    print(df[periodic_variable])
    # Get the cosinus and the sinus
    df[f"cos_{periodic_variable}"] = np.cos(df[periodic_variable])
    df[f"sin_{periodic_variable}"] = np.sin(df[periodic_variable])

    if drop_original_variable:
        return df.drop(columns=periodic_variable)
    return df


def reverse_periodic_variables(
    df, periodic_variable, modulo_variable, drop_periodic_variables=True
):
    """
    Using cos_{periodic_variable} and sin_{periodic_variable}, reverse the function generate_periodic_variables
    to get back the original value of periodic_variable.

    Parameters
    ----------
    df: pd.DataFrame
        contains at least the cos_{periodic_variable} and sin_{periodic_variable}
    periodic_variable: str
        name of the variable that is periodic
    modulo_variable: int
        the periodic_variable has a period of modulo_variable, that is to say that in the periodic_variable
        0 is equivalent to modulo_variable and that periodic_variable is between 0 and modulo_variable-1
    drop_periodic_variables: bool, optional
        if True the cos_{periodic_variable} and sin_{periodic_variable} are dropped from the dataframe.
        Default is True

    Returns
    ----------
    pd.DataFrame
        original DataFrame with the new added variable

    Examples:
    ---------
    >>> print(df)
    >>> print(def reverse_periodic_variables(df, "angle", 360, True)
    """
    # TODO Make it work ? the angles given are not right
    # Get the angle value from the cos and the sin
    df[periodic_variable] = np.arctan2(
        df[f"sin_{periodic_variable}"], df[f"cos_{periodic_variable}"]
    )
    print(df[periodic_variable])
    # Transform the angle into the variable
    df[periodic_variable] = df[periodic_variable] / (2 * np.pi / modulo_variable)

    if drop_periodic_variables:
        return df.drop(columns=[f"cos_{periodic_variable}", f"sin_{periodic_variable}"])
    return df


def get_temporal_variables(df, date_variable):
    """
    Get from a date variable new informations that can be used to create a model with a machine learning
    algorithm.
    The new features will be time, day (day of the week), month and year.

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
    >>> print(def get_temporal_variables(df, "date")
    """
    df[date_variable] = pd.to_datetime(df[date_variable])

    df["time"] = df[date_variable].dt.time
    df["day"] = df[date_variable].dt.day
    df["month"] = df[date_variable].dt.month
    df["year"] = df[date_variable].dt.year

    return df


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
    >>> print(def get_periodic_temporal_variable(df, "date")
    """
    temporal_df = get_temporal_variables(df, date_variable)

    temporal_df = generate_periodic_variables(
        temporal_df, periodic_variable="time", max_value=24, drop_original_variable=True
    )
    temporal_df = generate_periodic_variables(
        temporal_df, periodic_variable="day", max_value=7, drop_original_variable=True
    )
    temporal_df = generate_periodic_variables(
        temporal_df,
        periodic_variable="month",
        max_value=12,
        drop_original_variable=True,
    )

    return temporal_df


def reverse_get_periodic_temporal_variable(df, new_date_variable):
    pass


#################
# Main function #
#################

if __name__ == "__main__":
    # City test
    d = [
        {"ville": "Paris"},
        {"ville": "Lyon"},
        {"ville": "Saint-Étienne"},
        {"ville": "Vauxbuin"},
    ]
    df = pd.DataFrame(d, index=range(len(d)))
    print(df)
    print(city_encoding(df, "ville"))

    # Periodic test
    d = [
        {"month": 0},
        {"month": 1},
        {"month": 2},
        {"month": 3},
        {"month": 4},
        {"month": 5},
        {"month": 6},
        {"month": 7},
        {"month": 8},
        {"month": 9},
        {"month": 10},
        {"month": 11},
    ]
    df = pd.DataFrame(d, index=range(len(d)))
    print(df)
    df_periodic = generate_periodic_variables(df, "month", 12)
    print(df_periodic)
    df_periodic = reverse_periodic_variables(df_periodic, "month", 12)
    print(df_periodic)
