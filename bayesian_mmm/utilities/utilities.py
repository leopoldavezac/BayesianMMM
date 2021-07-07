from re import sub
from random import randint
from numpy import ndarray
from typing import Dict, List
import yaml
from pandas import DataFrame, read_csv, to_datetime

def trim(text: str) -> str:
    return "".join([char for char in text if char not in ["\n", "\t", " "]])

def remove_id_ref_from_plotly_html(html_str: str) -> str:

    p = r'[0-9a-z]{8}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{12}'
    return sub(p,"",html_str)

def remove_color_ref_from_ploty_html(html_str):

    p = r'\d{1,3}, \d{1,3}, \d{1,3}'
    return sub(p, "", html_str)


def generate_random_color() -> str:

    return "rgb(%d, %d, %d)" % (randint(0, 255), randint(0, 255), randint(0, 255))


def check_ndarray_is_vector(array: ndarray, var_nm: str) -> None:

    if not len(array.shape) == 1:
        raise ValueError("%s must be 1 dimensional" % var_nm)

def check_ndarray_is_matrix(array: ndarray, var_nm: str) -> None:

    if not len(array.shape) == 2:
        raise ValueError("%s must be 2 dimensional" % var_nm)

def check_ndarray_is_tensor(array: ndarray, var_nm: str) -> None:

    if not len(array.shape) == 3:
        raise ValueError("%s must be 3 dimensional" % var_nm)


def load_config(name: str) -> Dict:

    with open("config/%s.yaml" % name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config


def load_df(file_path: str, date_nm: str) -> DataFrame:

    df = read_csv(file_path)
    df[date_nm] = to_datetime(df[date_nm], format="%Y-%m-%d")
    df.set_index(date_nm, inplace=True)
    df.sort_index(inplace=True)

    return df


def split_train_test(df: DataFrame, nb_test_obs: int) -> List[DataFrame]:

    train = df.iloc[:-nb_test_obs].copy(deep=True)
    test = df.iloc[-nb_test_obs:].copy(deep=True)

    return [train, test]




