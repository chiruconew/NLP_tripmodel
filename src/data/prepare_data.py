import pandas as pd
from pandas import DataFrame


def read_sample() -> DataFrame:
    df = pd.read_csv('C:/Users/chiruco/Desktop/python/ProyPython/NLP_tripmodel/NLP_tripmodel/data/raw/reviews.csv')
    return df