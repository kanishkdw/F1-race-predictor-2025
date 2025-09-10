import fastf1
import pandas as pd
import os

from utils.data_loader import load_session, load_driver_mapping
from utils.preprocess import extract_sector_times
from utils.model import train_model

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Step 1: Fetch and save 2025 Australian GP qualifying data
qualifying_path = "data/qualifying_australia_2025.csv"
if not os.path.exists(qualifying_path) or os.path.getsize(qualifying_path) == 0:
    try:
        print("Loading 2025 Australian GP Qualifying Data...")
        session = fastf1.get_session(2025, "Australia", "Q")
        session.load()

        if not session.results.empty:
            qualifying_data = session.results[['Abbreviation', 'Position', 'Q1', 'Q2', 'Q3']]
            qualifying_data = qualifying_data.rename(columns={"Abbreviation": "DriverCode"})
            qualifying_data["QualifyingTime (s)"] = qualifying_data[["Q3", "Q2", "Q1"]].bfill(axis=1).iloc[:, 0].dt.total_seconds()
            qualifying_data = qualifying_data[["DriverCode", "Position", "QualifyingTime (s)"]]
            qualifying_data.to_csv(qualifying_path, index=False, encoding="utf-8")
            print(f"Qualifying data saved to '{qualifying_path}'")
        else:
            print("No qualifying data available (session likely hasn't happened yet).")
    except Exception as e:
        print(f"Error fetching qualifying data: {e}")

# Step 2: Load race session data from 2024 Australian GP
print("\nLoading 2024 Australian GP Race Data...")
session = load_session(2024, "Australia")
sector_times_2024, lap_averages_2024 = extract_sector_times(session)

# Step 3: Load driver mapping and 2025 qualifying data
driver_mapping = load_driver_mapping("driver_mapping.json")

try:
    qualifying_2025 = pd.read_csv(qualifying_path)

    if "DriverCode" not in qualifying_2025.columns:
        raise ValueError("Missing 'DriverCode' in qualifying CSV.")

    # Map DriverCode to full name
    qualifying_2025["Driver"] = qualifying_2025["DriverCode"].map(driver_mapping)

    # Fallback if mapping fails
    qualifying_2025["Driver"] = qualifying_2025["Driver"].fillna(qualifying_2025["DriverCode"])

    print("\n Successfully loaded qualifying data:")
    print(qualifying_2025.head())

except Exception as e:
    print(f"\n Failed to load 2025 qualifying data: {e}")
    qualifying_2025 = pd.DataFrame()

# Step 4: Proceed only if qualifying data is valid
if not qualifying_2025.empty:
    # Merge using full driver name
    merged = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")

    # Define feature set
    X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]

    # Define target variable using lap averages
    y_map = lap_averages_2024.set_index("Driver")["LapTime (s)"].to_dict()
    y = merged["Driver"].map(y_map)

    # ✅ Drop NaNs before training
    Xy = pd.concat([X, y.rename("LapTime")], axis=1).dropna()
    X = Xy.drop(columns=["LapTime"])
    y = Xy["LapTime"]

    print(f"\n Training on {len(X)} drivers after dropping NaNs.")

    # Train model
    print("\n Training prediction model...")
    model, error = train_model(X, y)

    # Predict race performance
    merged["PredictedRaceTime (s)"] = model.predict(
        merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
    )
    final_results = merged.sort_values(by="PredictedRaceTime (s)")

    # Save predictions
    os.makedirs("predictions", exist_ok=True)
    final_results.to_csv("predictions/prediction_australia_2025.csv", index=False)

    # Display results
    print("\n Predicted 2025 Australian GP Results\n")
    print(final_results[["Driver", "PredictedRaceTime (s)"]])
    print(f"\n Mean Absolute Error (MAE): {error:.2f} seconds")

    # Save MAE to file
    with open("predictions/mae_australia_2025.txt", "w") as f:
        f.write(str(round(error, 3)))

else:
    print("\n Prediction aborted: No valid qualifying data available.")