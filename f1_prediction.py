# -*- coding: utf-8 -*-
"""
F1 Race Position Prediction System (Backend) - Direct UI Output

This script sets up a Gradio application to serve predictions from the
trained F1 race prediction models. It loads data, trains models, and
provides a direct Gradio UI to get predictions for *all* drivers in a
selected race, along with race-specific performance metrics and a comparison table.
The output table will show the best model's predictions first.

To run this:
1. Save the code as a Python file (e.g., app_ui.py).
2. Install necessary libraries: pip install fastf1 pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm gradio
3. Run from your terminal: python app_ui.py
   - To potentially utilize GPU (if libraries like XGBoost/LightGBM are built with GPU support and drivers are installed):
     Depending on the library, you might need specific installation steps (e.g., for XGBoost with CUDA).
     The code itself doesn't explicitly manage GPU devices, it relies on the underlying libraries.

Gradio will start a local web server and provide a URL (usually http://127.0.0.1:7860).
"""

# @title 1. Setup and Installations (Ensure libraries are installed in your environment)
# !pip install fastf1 pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm gradio --quiet

import fastf1 as ff1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import logging
import gradio as gr # Import Gradio
import json # Used for potential debugging or structured data if needed, but main output is DataFrame/text

# Suppress warnings and configure logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('fastf1').setLevel(logging.ERROR)

# Configure FastF1 caching
cache_dir = './ff1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
ff1.Cache.enable_cache(cache_dir)

print("Libraries imported. FastF1 Cache Directory:", cache_dir)

# @title 2. Configuration
YEARS = [2021, 2022, 2023]
GRAND_PRIX_EXCLUDE = ['Emilia Romagna']
SESSION_NAME = 'R'
TEST_YEAR = 2023

FEATURE_COLS = ['GridPosition'] # Using only GridPosition as the feature
TARGET_COL = 'Position' # Original column name from FastF1 results

# Global variables to store data, preprocessor, models, and performance
combined_df = None
preprocessor = None
trained_models = {}
performance_metrics = {} # Performance on the TEST_YEAR data
feature_names_out = []
available_races = []


# @title 3. Data Loading and Feature Engineering Function
def get_race_data(year, gp_name, session_name='R'):
    """
    Loads race session data, including qualifying results, for a given year and GP.
    Engineers basic features like GridPosition.
    Returns a DataFrame with data for each driver in the race, including names and team.
    Ensures 'ActualPosition' column is present by renaming 'Position'.
    Handles cases where 'Position' might be missing in raw results.
    """
    try:
        session = ff1.get_session(year, gp_name, session_name)
        # Load only necessary data for efficiency
        session.load(laps=False, telemetry=False, weather=False, messages=False)

        if session.results is None or session.results.empty:
            print(f"No results found for {year} {gp_name} {session_name}. Skipping.")
            return None

        results_df = pd.DataFrame(session.results)

        # --- Check if 'Position' column exists ---
        if TARGET_COL not in results_df.columns:
            print(f"Missing essential column '{TARGET_COL}' in results for {year} {gp_name}. Skipping.")
            return None

        # --- Ensure other essential columns are present ---
        essential_cols_check = ['DriverNumber', 'GridPosition', 'Status']
        for col in essential_cols_check:
            if col not in results_df.columns:
                print(f"Missing essential column '{col}' in results for {year} {gp_name}. Skipping.")
                return None


        # --- Rename 'Position' to 'ActualPosition' early and handle non-numeric positions ---
        results_df[TARGET_COL] = pd.to_numeric(results_df[TARGET_COL], errors='coerce')
        max_pos_before_fill = results_df[TARGET_COL].max()
        results_df[TARGET_COL] = results_df[TARGET_COL].fillna(max_pos_before_fill + 1)
        results_df = results_df.rename(columns={TARGET_COL: 'ActualPosition'})


        # --- Add Driver Name and Team Info ---
        # Prioritize BroadcastName and TeamName from results if available
        # Otherwise, try to get from session.drivers
        temp_df = results_df.copy() # Start with results including ActualPosition

        # Try to get BroadcastName and TeamName from results_df if they exist
        if 'BroadcastName' not in temp_df.columns and 'BroadcastName' in results_df.columns:
             temp_df['BroadcastName'] = results_df['BroadcastName']
        if 'TeamName' not in temp_df.columns and 'TeamName' in results_df.columns:
             temp_df['TeamName'] = results_df['TeamName']


        # If BroadcastName or TeamName are still missing, try session.drivers
        if ('BroadcastName' not in temp_df.columns or temp_df['BroadcastName'].isnull().all()) or \
           ('TeamName' not in temp_df.columns or temp_df['TeamName'].isnull().all()):

            if hasattr(session, 'drivers') and session.drivers:
                driver_info = pd.DataFrame.from_dict(session.drivers, orient='index')
                # Ensure 'BroadcastName' and 'Team' columns exist in driver_info
                if 'BroadcastName' in driver_info.columns and 'Team' in driver_info.columns:
                    driver_info = driver_info[['BroadcastName', 'Team']].rename(columns={'Team': 'TeamName_driver_info'})
                    # Merge with temp_df on DriverNumber
                    temp_df = temp_df.merge(driver_info, left_on='DriverNumber', right_index=True, how='left')

                    # Fill missing BroadcastName/TeamName from merged driver_info
                    if 'BroadcastName' not in temp_df.columns or temp_df['BroadcastName'].isnull().all():
                         if 'BroadcastName' in temp_df.columns and 'BroadcastName_driver_info' in temp_df.columns: # Check existence before fillna
                              temp_df['BroadcastName'] = temp_df['BroadcastName'].fillna(temp_df['BroadcastName_driver_info'])
                    if 'TeamName' not in temp_df.columns or temp_df['TeamName'].isnull().all():
                         if 'TeamName' in temp_df.columns and 'TeamName_driver_info' in temp_df.columns: # Check existence before fillna
                              temp_df['TeamName'] = temp_df['TeamName'].fillna(temp_df['TeamName_driver_info'])

                    # Drop the merged columns from driver_info
                    temp_df = temp_df.drop(columns=[col for col in temp_df.columns if col.endswith('_driver_info')])


        # Ensure BroadcastName and TeamName columns exist, even if empty
        if 'BroadcastName' not in temp_df.columns:
             temp_df['BroadcastName'] = None
        if 'TeamName' not in temp_df.columns:
             temp_df['TeamName'] = None


        # --- Final DataFrame Structure ---
        final_df = temp_df.copy()
        final_df['Year'] = year
        final_df['RaceName'] = gp_name
        final_df['RaceId'] = f"{year}_{gp_name}"

        # Split BroadcastName into First and Last Name
        if 'BroadcastName' in final_df.columns and final_df['BroadcastName'] is not None:
             final_df[['DriverFirstName', 'DriverLastName']] = final_df['BroadcastName'].astype(str).str.split(' ', n=1, expand=True)
             final_df = final_df.drop(columns=['BroadcastName'])
        else:
             final_df['DriverFirstName'] = 'Unknown'
             final_df['DriverLastName'] = 'Driver'


        # Define the final set of columns to return, ensuring 'ActualPosition' is included
        final_cols_to_return = ['RaceId', 'Year', 'RaceName', 'DriverNumber', 'DriverFirstName', 'DriverLastName', 'TeamName', 'GridPosition', 'Status', 'ActualPosition']
        # Ensure all final columns exist before selecting (fill with None/NaN if still missing)
        for col in final_cols_to_return:
            if col not in final_df.columns:
                final_df[col] = None if col in ['DriverFirstName', 'DriverLastName', 'TeamName', 'Status'] else np.nan # Assign appropriate default

        # Ensure correct data types for critical columns
        final_df['GridPosition'] = pd.to_numeric(final_df['GridPosition'], errors='coerce')
        final_df['ActualPosition'] = pd.to_numeric(final_df['ActualPosition'], errors='coerce')


        return final_df[final_cols_to_return]

    except Exception as e:
        print(f"Error loading data for {year} {gp_name}: {e}")
        return None

# @title 4. Load and Preprocess Data (Runs once on startup)
def load_and_preprocess_data():
    """Loads data, preprocesses it, and trains models."""
    global combined_df, preprocessor, trained_models, performance_metrics, feature_names_out, available_races

    all_race_data = []
    processed_races = set()

    print("\n--- Loading Data ---")
    for year in YEARS:
        print(f"Processing Year: {year}")
        schedule = ff1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        races_in_year = schedule['EventName'].unique()

        for gp_name in races_in_year:
            race_id = f"{year}_{gp_name}"
            if gp_name in GRAND_PRIX_EXCLUDE or race_id in processed_races:
                print(f"Skipping {year} {gp_name} (Excluded or already processed).")
                continue

            print(f"Fetching data for: {year} {gp_name}")
            race_df = get_race_data(year, gp_name, SESSION_NAME)

            if race_df is not None and not race_df.empty:
                all_race_data.append(race_df)
                processed_races.add(race_id)
            elif race_df is not None and race_df.empty:
                 print(f"Dataframe for {year} {gp_name} is empty after processing. Skipping.")


    if not all_race_data:
        print("\nNo data loaded. Cannot train models.")
        return

    combined_df = pd.concat(all_race_data, ignore_index=True)
    print(f"\n--- Data Loading Complete ---")
    print(f"Total races loaded: {combined_df['RaceId'].nunique()}")
    print(f"Total driver results loaded: {len(combined_df)}")

    # Store available races for the Gradio dropdown
    available_races = combined_df['RaceId'].unique().tolist()


    # --- Data Split (Time-Based or GroupShuffleSplit) ---
    train_df = combined_df[combined_df['Year'] < TEST_YEAR].copy()
    test_df = combined_df[combined_df['Year'] == TEST_YEAR].copy()

    if train_df.empty or test_df.empty:
        print(f"Using GroupShuffleSplit as only data for year(s) <= {TEST_YEAR} loaded.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        # Ensure groups are valid (not all same or too few)
        if combined_df['RaceId'].nunique() > 1 and len(combined_df) > 1:
            train_idxs, test_idxs = next(splitter.split(combined_df, groups=combined_df['RaceId']))
            train_df = combined_df.iloc[train_idxs].copy()
            test_df = combined_df.iloc[test_idxs].copy()
        else:
             print("Cannot perform GroupShuffleSplit with insufficient data.")
             return


    if train_df.empty or test_df.empty:
         print("Train or Test DataFrame is empty after splitting. Cannot proceed with training.")
         return

    # Ensure 'ActualPosition' is the target for training/testing
    # Check if 'ActualPosition' exists in train_df and test_df
    if 'ActualPosition' not in train_df.columns or 'ActualPosition' not in test_df.columns:
        print("Error: 'ActualPosition' column missing in train or test data after splitting. Cannot train models.")
        return

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['ActualPosition'] # Use 'ActualPosition' as target
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['ActualPosition'] # Use 'ActualPosition' as target


    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Training Races: {train_df['RaceId'].nunique()}, Testing Races: {test_df['RaceId'].nunique()}")

    # --- Preprocessing Pipeline ---
    numerical_features = FEATURE_COLS # In this simple case, all features are numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), # Impute missing values
                ('scaler', StandardScaler()) # Scale features
            ]), numerical_features)
        ],
        remainder='passthrough'
    )

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names_out = numerical_features # Store feature names

    print("\nPreprocessing complete.")
    print("Processed training features shape:", X_train_processed.shape)

    # @title 5. Model Training
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(objective='regression_l1', n_estimators=100, random_state=42, n_jobs=-1)
    }

    print("\n--- Training Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model

        # Evaluate performance on the test set (TEST_YEAR)
        y_pred = model.predict(X_test_processed)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        performance_metrics[name] = {'MAE': round(mae, 3), 'R2': round(r2, 3)} # Store rounded metrics

        print(f"{name} - MAE: {mae:.3f}, R2: {r2:.3f}")

    print("\n--- Model Training Complete ---")

# Load and preprocess data when the script starts
load_and_preprocess_data()

# @title 6. Prediction Function for Gradio (Predicting for all drivers and returning DataFrame/Text)
def predict_and_display_race_positions(race_id: str) -> tuple[str, pd.DataFrame]:
    """
    Predicts the finishing positions for all drivers in a specific race
    using all trained models. Returns a formatted summary string and a
    DataFrame for display in Gradio.

    Args:
        race_id: The unique ID of the race (e.g., '2023_Bahrain Grand Prix').

    Returns:
        A tuple containing:
        - A formatted string with race summary and performance.
        - A pandas DataFrame with detailed driver predictions.
        Returns error messages/empty DataFrame if data is not available or error occurs.
    """
    if combined_df is None or preprocessor is None or not trained_models:
        error_msg = "Models not trained or data not loaded. Check backend logs."
        return error_msg, pd.DataFrame({"Error": [error_msg]})

    # Find all driver data for the selected race
    race_data = combined_df[combined_df['RaceId'] == race_id].copy()

    if race_data.empty:
        error_msg = f"No data found for race: {race_id}. Please select from available options."
        return error_msg, pd.DataFrame({"Error": [error_msg]})

    # Sort by Grid Position for a more intuitive output order
    race_data = race_data.sort_values(by='GridPosition').reset_index(drop=True)

    # Extract the feature(s) for prediction for all drivers in the race
    X_predict = race_data[FEATURE_COLS]

    # Preprocess the data using the fitted preprocessor
    try:
        X_predict_processed = preprocessor.transform(X_predict)
    except Exception as e:
        error_msg = f"Error during preprocessing for prediction: {e}"
        return error_msg, pd.DataFrame({"Error": [error_msg]})

    # Use the best performing model (based on overall test MAE) for the race winner prediction
    best_model_name = min(performance_metrics, key=lambda k: performance_metrics[k]['MAE']) if performance_metrics else list(trained_models.keys())[0]
    best_model = trained_models.get(best_model_name)

    if best_model is None:
         error_msg = "Best model not found or trained."
         return error_msg, pd.DataFrame({"Error": [error_msg]})

    # Get predictions from the best model to determine the predicted winner
    best_model_predictions_raw = best_model.predict(X_predict_processed)
    best_model_predictions_rounded = np.round(best_model_predictions_raw)

    # Find the predicted winner based on the lowest rounded predicted position from the best model
    predicted_winner_info = None
    if len(best_model_predictions_rounded) > 0:
        min_predicted_pos = np.min(best_model_predictions_rounded)
        # Find indices where predicted position equals the minimum
        predicted_winner_indices = np.where(best_model_predictions_rounded == min_predicted_pos)[0]
        # If multiple drivers have the same predicted minimum position, pick the one with the best grid position
        if len(predicted_winner_indices) > 1:
             # Get the subset of race_data for these potential winners
             potential_winners_data = race_data.iloc[predicted_winner_indices]
             # Find the index (within potential_winners_data) of the driver with the minimum grid position
             best_grid_index_in_potential = potential_winners_data['GridPosition'].idxmin()
             # Get the original index in race_data
             predicted_winner_index = best_grid_index_in_potential
        else:
             predicted_winner_index = predicted_winner_indices[0]


        predicted_winner_driver_number = race_data.iloc[predicted_winner_index]['DriverNumber']
        predicted_winner_first_name = race_data.iloc[predicted_winner_index]['DriverFirstName']
        predicted_winner_last_name = race_data.iloc[predicted_winner_index]['DriverLastName']
        predicted_winner_team = race_data.iloc[predicted_winner_index]['TeamName']
        predicted_winner_info = {
            "driver_number": predicted_winner_driver_number,
            "first_name": predicted_winner_first_name,
            "last_name": predicted_winner_last_name,
            "team": predicted_winner_team
        }

    # Calculate race-specific performance metrics for the best model
    # Only consider drivers who finished (Status is 'Finished') for accurate MAE and exact predictions
    race_performance_details = {
        "mae": None,
        "exact_predictions_count": 0,
        "total_finished_drivers": 0,
        "exact_predictions_percentage": 0
    }
    # Check if 'ActualPosition' exists and is numeric before filtering
    if 'ActualPosition' in race_data.columns and pd.api.types.is_numeric_dtype(race_data['ActualPosition']):
        finished_drivers_data = race_data[race_data['Status'] == 'Finished'].copy()
        # Ensure finished_drivers_data has 'ActualPosition' and it's numeric
        if not finished_drivers_data.empty and 'ActualPosition' in finished_drivers_data.columns and pd.api.types.is_numeric_dtype(finished_drivers_data['ActualPosition']):
            # Get the indices of finished drivers in the original race_data DataFrame
            finished_drivers_original_indices = finished_drivers_data.index
            # Get the corresponding predictions using these original indices
            # Use .loc to ensure index alignment
            predicted_positions_finished_raw = pd.Series(best_model_predictions_raw, index=race_data.index).loc[finished_drivers_original_indices]
            predicted_positions_finished_rounded = np.round(predicted_positions_finished_raw)

            actual_positions_finished = finished_drivers_data['ActualPosition'] # Use ActualPosition column


            # Ensure actual and predicted series have the same index before calculating metrics
            if not actual_positions_finished.empty and not predicted_positions_finished_raw.empty and actual_positions_finished.index.equals(predicted_positions_finished_raw.index):
                race_mae = mean_absolute_error(actual_positions_finished, predicted_positions_finished_raw)
                exact_predictions_count = np.sum(predicted_positions_finished_rounded == actual_positions_finished)
                total_finished_drivers = len(finished_drivers_data)
                exact_predictions_percentage = (exact_predictions_count / total_finished_drivers) * 100 if total_finished_drivers > 0 else 0

                race_performance_details = {
                    "mae": round(race_mae, 3),
                    "exact_predictions_count": int(exact_predictions_count),
                    "total_finished_drivers": int(total_finished_drivers),
                    "exact_predictions_percentage": round(exact_predictions_percentage, 1)
                }
            else:
                 print(f"Warning: Index mismatch or empty data for race performance calculation for {race_id}. This might happen if no drivers finished.")


    # Prepare the DataFrame for display
    display_df_data = []

    # Add predictions from all models for each driver
    for index, row in race_data.iterrows():
        driver_data = {
            "Driver First Name": row['DriverFirstName'],
            "Driver Last Name": row['DriverLastName'],
            "Team": row['TeamName'],
            "Grid": float(row['GridPosition']) if pd.notna(row['GridPosition']) else None,
            "Actual Position": float(row['ActualPosition']) if pd.notna(row['ActualPosition']) else None,
            "Status": row['Status']
        }

        for name, model in trained_models.items():
            try:
                # Get the prediction for this specific driver
                # Use the index in the potentially re-indexed race_data to get the corresponding processed feature row
                driver_index_in_processed_data = race_data.index.get_loc(index)
                prediction_raw = model.predict(X_predict_processed[driver_index_in_processed_data].reshape(1, -1))[0]
                prediction_rounded = round(prediction_raw)

                # Calculate prediction error only if ActualPosition is available and is a number
                prediction_error = 'N/A'
                if driver_data["Actual Position"] is not None and isinstance(driver_data["Actual Position"], (int, float)):
                    prediction_error = int(prediction_rounded - driver_data["Actual Position"])


                driver_data[f'Predicted Position (Raw) ({name})'] = round(float(prediction_raw), 3)
                driver_data[f'Predicted Position (Rounded) ({name})'] = int(prediction_rounded)
                driver_data[f'Prediction Error ({name})'] = prediction_error


            except Exception as e:
                driver_data[f'Predicted Position (Raw) ({name})'] = f"Error: {e}"
                driver_data[f'Predicted Position (Rounded) ({name})'] = "Error"
                driver_data[f'Prediction Error ({name})'] = "Error"

        display_df_data.append(driver_data)

    # Create the DataFrame
    display_df = pd.DataFrame(display_df_data)

    # --- Reorder columns to show best model first ---
    standard_cols = ["Driver First Name", "Driver Last Name", "Team", "Grid", "Actual Position", "Status"]
    best_model_cols = [
        f'Predicted Position (Raw) ({best_model_name})',
        f'Predicted Position (Rounded) ({best_model_name})',
        f'Prediction Error ({best_model_name})'
    ]
    other_model_cols = []
    for name in trained_models.keys():
        if name != best_model_name:
            other_model_cols.extend([
                f'Predicted Position (Raw) ({name})',
                f'Predicted Position (Rounded) ({name})',
                f'Prediction Error ({name})'
            ])

    # Combine all columns in the desired order
    ordered_cols = standard_cols + best_model_cols + other_model_cols

    # Ensure all ordered_cols actually exist in the DataFrame before selecting
    # This handles cases where a model might have failed to produce predictions for a driver
    valid_ordered_cols = [col for col in ordered_cols if col in display_df.columns]

    # Select and reorder columns
    display_df = display_df[valid_ordered_cols]


    # Format the race summary string
    summary_text = f"## {race_data['RaceName'].iloc[0]} - Race Results vs Predictions\n\n"
    if predicted_winner_info:
        summary_text += f"**Race Winner:** {predicted_winner_info['first_name']} {predicted_winner_info['last_name']} ({predicted_winner_info['team']})\n\n"
    else:
         summary_text += "**Race Winner:** N/A\n\n"

    summary_text += "**Prediction Performance (Best Model for this Race):**\n"
    summary_text += f"- Mean Absolute Error (MAE): {race_performance_details['mae'] if race_performance_details['mae'] is not None else 'N/A'} positions (on {race_performance_details['total_finished_drivers']} drivers with valid actual/predicted positions)\n"
    summary_text += f"- Exact Position Predictions: {race_performance_details['exact_predictions_count']} out of {race_performance_details['total_finished_drivers']} ({race_performance_details['exact_predictions_percentage'] if race_performance_details['exact_predictions_percentage'] is not None else 'N/A'}%)\n\n"
    summary_text += "### Race Results Comparison"


    return summary_text, display_df


# @title 7. Gradio Interface Setup

# Create Gradio components for input and output outside the Blocks context first
race_dropdown = gr.Dropdown(choices=available_races, label="Select Race")

# Outputs will be a Markdown component for the summary and a Dataframe for the table
summary_output = gr.Markdown()
table_output = gr.Dataframe()

# Overall performance markdown
overall_performance_md = gr.Markdown()


# Create the Gradio app using gr.Blocks to provide the necessary context
if available_races: # Only create app if data was loaded
    with gr.Blocks(title="F1 Race Position Prediction System") as demo: # Set title for the browser tab
        gr.Markdown("# F1 Race Position Prediction System")
        gr.Markdown("Select a past race to see the model's predicted finishing positions compared to the actual results. Lower 'Actual Position' is better.")

        # Arrange inputs in a row
        with gr.Row():
            race_dropdown.render() # Render the race dropdown

        # Add buttons in a row
        with gr.Row():
             clear_button = gr.Button("Clear")
             predict_button = gr.Button("Submit")

        # Arrange outputs in a column
        with gr.Column():
            summary_output.render() # Render the summary markdown
            table_output.render() # Render the dataframe
            gr.Markdown("### Overall Model Performance (on Test Data)") # Heading for overall performance
            overall_performance_md.render() # Render the overall performance markdown


        # Link the predict button click to the prediction function
        predict_button.click(
            fn=predict_and_display_race_positions,
            inputs=[race_dropdown], # Only race_dropdown is the input
            outputs=[summary_output, table_output] # Output to both components
        )

        # Link the clear button click to a function that clears outputs and resets dropdown
        def clear_outputs():
            # Clear both output components and reset the dropdown value
            return "", pd.DataFrame(), gr.Dropdown(value=None, choices=available_races)

        clear_button.click(
            fn=clear_outputs,
            inputs=[], # No inputs for clear
            outputs=[summary_output, table_output, race_dropdown] # Clear outputs and reset dropdown
        )

        # Display overall model performance on load
        # Format the overall performance metrics into a Markdown string
        overall_perf_text = "Model | MAE | R2\n---|---|---\n"
        for model_name, metrics in performance_metrics.items():
             overall_perf_text += f"{model_name} | {metrics.get('MAE', 'N/A')} | {metrics.get('R2', 'N/A')}\n"

        overall_performance_md.value = overall_perf_text # Set the initial value of the markdown component


    # Launch the Gradio app using the Blocks object
    print("\n--- Launching Gradio App ---")
    print("Data loading and model training complete. Gradio app is starting...")
    print("Waiting for Gradio to launch...")

    try:
        # Launch the Blocks app
        demo.launch(share=False) # Set share=True for a public URL
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        print("Please check if the port is already in use or if there are other issues.")
        print("You might need to try a different port using: demo.launch(server_port=XXXX)")

else:
    print("Gradio app not created because no data was loaded.")
