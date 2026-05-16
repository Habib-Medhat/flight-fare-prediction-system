import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from flight_utils import (
    CATEGORICAL_FEATURES,
    FARE_COLUMNS,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    TARGET_COLUMN,
    cap_outliers_iqr,
    load_flight_data,
    prepare_flight_data,
)


RANDOM_STATE = 42


def print_dataset_overview(df: pd.DataFrame) -> None:
    print(df.head())
    print(df.info())
    print("Categorical Columns:", df.select_dtypes(include=["object"]).columns.tolist())
    print("Numerical Columns:", df.select_dtypes(include=["int64", "float64"]).columns.tolist())
    print("Number of duplicates:", df.duplicated().sum())
    print("Missing values:")
    print(df.isnull().sum())


def plot_fare_distributions(df: pd.DataFrame, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, len(FARE_COLUMNS), figsize=(12, 6))

    for axis, column in zip(axes, FARE_COLUMNS):
        sns.boxplot(y=df[column], color="skyblue", ax=axis)
        axis.set_title(f"{column}{title_suffix}")
        axis.set_ylabel("Fare (BDT)")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, len(FARE_COLUMNS), figsize=(18, 5))

    for axis, column in zip(axes, FARE_COLUMNS):
        sns.histplot(df[column], kde=True, color="skyblue", ax=axis)
        axis.set_title(f"{column}{title_suffix}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def print_outlier_counts(df: pd.DataFrame) -> None:
    for column in FARE_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"{column}: {len(outliers)} outliers")


def run_pca(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    X_combined = pd.DataFrame(encoded_data)
    X_combined.columns = X_combined.columns.astype(str)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    return pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])


def print_hypotheses() -> None:
    hypotheses = [
        "H1: The dataset contains duplicate records that should be removed to ensure accurate analysis.",
        "H2: Missing values, if present in any feature, may distort analysis and model performance.",
        "H3: Outliers exist in the fare-related columns and may negatively affect model performance.",
        "H4: Capping outliers in fare columns will improve data quality.",
        "H5: The Total Fare is strongly correlated with both the Base Fare and the Tax & Surcharge.",
        "H6: The Base Fare contributes more to the Total Fare than the Tax & Surcharge.",
        "H7: The departure day, month, hour, and weekday influence flight prices.",
        "H8: Weekend or holiday-season flights are more expensive.",
        "H9: Airline choice significantly affects the total flight fare.",
        "H10: Stopovers, booking source, and travel class contribute to fare variation.",
        "H11: PCA can reduce encoded categorical and temporal features into useful components.",
        "H12: The top PCA components can reveal groups of flights with similar characteristics.",
    ]

    for hypothesis in hypotheses:
        print(hypothesis)


def print_statistical_tests(df: pd.DataFrame) -> None:
    for column in NUMERIC_FEATURES:
        corr, p_value = pearsonr(df[column], df[TARGET_COLUMN])
        print(f"Correlation between {column} and Total Fare: r = {corr:.4f}, p = {p_value:.5f}")

    categorical_tests = ["Airline", "Stopovers", "Class", "Booking Source", "Seasonality"]
    time_tests = ["Departure_Day", "Departure_Month", "Departure_Hour", "Departure_Weekday"]

    for column in [*categorical_tests, *time_tests]:
        groups = [group[TARGET_COLUMN].values for _, group in df.groupby(column)]
        f_stat, p_value = f_oneway(*groups)
        print(f"ANOVA for {column}: F = {f_stat:.2f}, p = {p_value:.5f}")


def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_cat = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    X = np.concatenate([encoded_cat, df[NUMERIC_FEATURES].values], axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2 Score": r2_score(y_test, y_pred),
        }

    return pd.DataFrame(results).T


def main() -> None:
    raw_df = load_flight_data()
    flight_df = prepare_flight_data(raw_df)
    flight_df.to_csv(PROJECT_DIR / "Flight_Price_Dataset_Updated.csv", index=False)

    print_dataset_overview(flight_df)
    plot_fare_distributions(flight_df)
    print_outlier_counts(flight_df)

    cleaned_df = cap_outliers_iqr(flight_df)
    plot_fare_distributions(cleaned_df, title_suffix=" (Outliers Capped)")

    print(run_pca(cleaned_df).head())
    print_hypotheses()
    print_statistical_tests(cleaned_df)
    print(evaluate_models(cleaned_df))


if __name__ == "__main__":
    main()
