import requests
import yaml
import pandas as pd
from config import RESPONSE_YAML_URL, DIA_PARQUET_PATH

def load_responses():
    response = requests.get(RESPONSE_YAML_URL)
    response.raise_for_status()
    data = yaml.safe_load(response.text)
    return pd.DataFrame(data)

def load_external_dataset():
    return pd.read_parquet(DIA_PARQUET_PATH)
