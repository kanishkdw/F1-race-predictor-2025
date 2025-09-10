import fastf1
import pandas as pd
import json

fastf1.Cache.enable_cache("f1_cache")

def load_session(year, gp, session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session

def load_driver_mapping(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def load_qualifying_data(filepath, driver_mapping):
    df = pd.read_csv(filepath)
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    return df
