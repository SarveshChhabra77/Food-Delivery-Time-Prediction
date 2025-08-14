import pandas as pd


## I choose multiple dtypes so that i can add flexibility

SCHEMA = {
    "Order_ID": [pd.StringDtype(), "object", str],
    "Distance_km": [float, "float64"],
    "Weather": [pd.StringDtype(), "object", str],
    "Traffic_Level": [pd.StringDtype(), "object", str],
    "Time_of_Day": [pd.StringDtype(), "object", str],
    "Vehicle_Type": [pd.StringDtype(), "object", str],
    "Preparation_Time_min": [float, "float64"],
    "Courier_Experience_yrs": [float, "float64"],
    "Delivery_Time_min": [float, "float64"]
}