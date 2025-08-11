import pandas as pd
import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df




def driver_historical_completed_bookings(df: pd.DataFrame, target_col: str = "is_completed", k_features: int = 15) -> pd.DataFrame:
    """
    Perform EDA, feature engineering, and feature selection on merged dataset.

    Parameters:
    - df: Merged dataset containing both booking and driver-level fields
    - target_col: Target variable name for feature selection
    - k_features: Number of top features to keep

    Returns:
    - Processed DataFrame with selected features and target (if target exists)
    """

    # --- 1. Detect if already merged ---
    required_booking_cols = {"trip_distance", "pickup_latitude", "pickup_longitude"}
    required_driver_cols = {"driver_latitude", "driver_longitude", "driver_gps_accuracy"}
    if not (required_booking_cols & set(df.columns) and required_driver_cols & set(df.columns)):
        raise ValueError("Dataset does not contain both booking and driver features. Merge step required.")

    # --- 2. EDA ---
    print("\n===== Missing Values =====")
    print(df.isnull().sum())

    print("\n===== Data Types =====")
    print(df.dtypes)

    print("\n===== Basic Statistics =====")
    print(df.describe())

    if target_col in df.columns:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.countplot(x=df[target_col])
        plt.title("Target Variable Distribution")
        plt.show()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    # --- 3. Calculate driver-level booking stats ---

    total_bookings_per_driver = (
        df.groupby('driver_id')
        .size()
        .reset_index(name='total_bookings')
    )

    target_exists = target_col in df.columns
    participant_status_exists = 'participant_status' in df.columns

    if target_exists:
        completed_bookings_per_driver = (
            df.groupby('driver_id')
            .apply(lambda g: (g[target_col] == 1).sum())
            .reset_index(name='completed_bookings')
        )
    else:
        completed_bookings_per_driver = pd.DataFrame({
            'driver_id': total_bookings_per_driver['driver_id'],
            'completed_bookings': 0
        })

    if participant_status_exists:
        accepted_bookings_per_driver = (
            df.groupby('driver_id')
            .apply(lambda x: (x['participant_status'] == 'ACCEPTED').sum())
            .reset_index(name='accepted_bookings')
        )

        cancelled_bookings_per_driver = (
            df.groupby('driver_id')
            .apply(lambda x: (x['participant_status'] == 'REJECTED').sum())
            .reset_index(name='cancelled_bookings')
        )
    else:
        accepted_bookings_per_driver = pd.DataFrame({
            'driver_id': total_bookings_per_driver['driver_id'],
            'accepted_bookings': 0
        })

        cancelled_bookings_per_driver = pd.DataFrame({
            'driver_id': total_bookings_per_driver['driver_id'],
            'cancelled_bookings': 0
        })

    # Merge all stats
    driver_stats = (
        total_bookings_per_driver
        .merge(completed_bookings_per_driver, on='driver_id', how='left')
        .merge(accepted_bookings_per_driver, on='driver_id', how='left')
        .merge(cancelled_bookings_per_driver, on='driver_id', how='left')
    )

    # Fill missing values with 0
    for col in ['total_bookings', 'completed_bookings', 'accepted_bookings', 'cancelled_bookings']:
        driver_stats[col] = driver_stats[col].fillna(0)

    # Feature engineering: calculate ratios safely
    feature_defs = {
        "driver_cancellation_rate": lambda x: x["cancelled_bookings"] / (x["total_bookings"] + 1e-5),
        "acceptance_rate": lambda x: x["accepted_bookings"] / (x["total_bookings"] + 1e-5),
        "driver_completed_ratio": lambda x: x["completed_bookings"] / (x["total_bookings"] + 1e-5),
    }

    for col_name, func in feature_defs.items():
        driver_stats[col_name] = func(driver_stats)

    # Select only driver_id + ratio columns
    ratios_to_merge = ["driver_id"] + list(feature_defs.keys())
    driver_ratios = driver_stats[ratios_to_merge]

    # Merge ratios back to original df
    df = df.merge(driver_ratios, on="driver_id", how="left")

    # Fill missing ratio values with 0
    ratio_cols = list(feature_defs.keys())
    df[ratio_cols] = df[ratio_cols].fillna(0)

    # --- 5. Feature Selection ---

    # Separate features and target if available
    if target_exists:
        X = df.drop(columns=[target_col], errors='ignore').select_dtypes(include="number")
        y = df[target_col]
    else:
        X = df.select_dtypes(include="number")
        y = None

    # Scale features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if target_exists:  # TRAINING MODE
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k=5)
        selector.fit(X_scaled, y)

        selected_features = X_scaled.columns[selector.get_support()].tolist()
        print(f"\nSelected Top {len(selected_features)} Features (ANOVA): {selected_features}")

        # Save the selected features list for prediction step
        pd.Series(selected_features).to_csv("selected_features.csv", index=False, header=False)

        # Return selected features + target
        return pd.concat([X_scaled[selected_features], y.reset_index(drop=True), df['order_id'].reset_index(drop=True)], axis=1)

    else:  # PREDICTION MODE
        # Load the feature list saved during training
        try:
            selected_features = pd.read_csv("selected_features.csv").squeeze().tolist()
        except FileNotFoundError:
            raise ValueError("Selected features file not found. Please train the model first.")

        # Only keep features that exist in the dataset (safe check)
        if "trip_distance" not in selected_features:
                selected_features.append("trip_distance")
        selected_features = [f for f in selected_features if f in X_scaled.columns]

        # Return scaled features limited to selected features
        id_cols = ['driver_id', 'order_id']
        df_ids = df[id_cols].reset_index(drop=True)
        result_df = pd.concat([df_ids, X_scaled[selected_features].reset_index(drop=True)], axis=1)
        return result_df