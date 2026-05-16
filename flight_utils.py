from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "bangladesh_flight_prices_dataset.csv"

DATE_COLUMNS = ["Departure Date & Time", "Arrival Date & Time"]
DROP_COLUMNS = ["Source Name", "Destination Name"]
FARE_COLUMNS = ["Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)"]
TIME_FEATURES = [
    "Departure_Day",
    "Departure_Month",
    "Departure_Hour",
    "Departure_Weekday",
]
CATEGORICAL_FEATURES = [
    "Airline",
    "Source",
    "Destination",
    "Stopovers",
    "Aircraft Type",
    "Class",
    "Booking Source",
    "Seasonality",
    *TIME_FEATURES,
]
NUMERIC_FEATURES = ["Base Fare (BDT)", "Tax & Surcharge (BDT)"]
TARGET_COLUMN = "Total Fare (BDT)"


def load_flight_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the flight dataset using an absolute path relative to this project."""
    return pd.read_csv(path)


def prepare_flight_data(df: pd.DataFrame, drop_airport_names: bool = True) -> pd.DataFrame:
    """Clean the dataset and add reusable date/time features."""
    cleaned = df.copy()

    for column in DATE_COLUMNS:
        if column in cleaned:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    departure = cleaned["Departure Date & Time"]
    cleaned["Departure_Day"] = departure.dt.day
    cleaned["Departure_Month"] = departure.dt.month
    cleaned["Departure_Hour"] = departure.dt.hour
    cleaned["Departure_Weekday"] = departure.dt.weekday

    if drop_airport_names:
        cleaned = cleaned.drop(columns=[col for col in DROP_COLUMNS if col in cleaned])

    return cleaned.drop_duplicates().dropna().reset_index(drop=True)


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str] = FARE_COLUMNS) -> pd.DataFrame:
    """Cap numeric outliers with the IQR rule instead of deleting rows."""
    capped = df.copy()

    for column in columns:
        q1 = capped[column].quantile(0.25)
        q3 = capped[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        capped[column] = capped[column].clip(lower=lower_bound, upper=upper_bound)

    return capped


def build_feature_matrix(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    """Encode model features and return X, y, and the fitted encoder."""
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_categories = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    else:
        encoded_categories = encoder.transform(df[CATEGORICAL_FEATURES])

    X = np.concatenate([encoded_categories, df[NUMERIC_FEATURES].to_numpy()], axis=1)
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df else None
    return X, y, encoder


def train_fare_model(df: pd.DataFrame | None = None):
    """Train the fare prediction model used by the API and analysis script."""
    if df is None:
        df = prepare_flight_data(load_flight_data())

    model_df = cap_outliers_iqr(df)
    X, y, encoder = build_feature_matrix(model_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    return model, encoder, (X_test, y_test)


def make_prediction_frame(data) -> pd.DataFrame:
    """Convert API input into the exact feature names used by the encoder."""
    return pd.DataFrame(
        [
            {
                "Airline": data.Airline,
                "Source": data.Source,
                "Destination": data.Destination,
                "Stopovers": data.Stopovers,
                "Aircraft Type": data.Aircraft_Type,
                "Class": data.Class,
                "Booking Source": data.Booking_Source,
                "Seasonality": data.Seasonality,
                "Departure_Day": data.Departure_Day,
                "Departure_Month": data.Departure_Month,
                "Departure_Hour": data.Departure_Hour,
                "Departure_Weekday": data.Departure_Weekday,
                "Base Fare (BDT)": data.Base_Fare,
                "Tax & Surcharge (BDT)": data.Tax_Surcharge,
            }
        ]
    )
