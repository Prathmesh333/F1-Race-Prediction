# F1 Race Prediction - Local Version with Gradio UI
# Save this as f1_prediction.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import gradio as gr
import warnings

# Ignore specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Configuration ---
# Determine script directory robustly
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Handle cases where __file__ is not defined (e.g., interactive environments)
    current_dir = os.getcwd()

data_dir = os.path.join(current_dir, 'data')
models_dir = os.path.join(current_dir, 'models')

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Data Loading ---

def load_f1_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load F1 datasets from CSV files in the specified directory.
    Includes the newly added sprint_results.csv.
    """
    dataframes = {}
    expected_files = [
        'circuits.csv',
        'constructor_results.csv',
        'constructor_standings.csv',
        'constructors.csv',
        'driver_standings.csv',
        'drivers.csv',
        'lap_times.csv',
        'pit_stops.csv',
        'qualifying.csv',
        'races.csv',
        'results.csv',
        'seasons.csv',
        'status.csv',
        'sprint_results.csv'  # Added sprint results
    ]

    print(f"Looking for data files in: {data_dir}")
    existing_files = os.listdir(data_dir)

    for file in expected_files:
        if file in existing_files:
            file_path = os.path.join(data_dir, file)
            table_name = file.split('.')[0]
            try:
                dataframes[table_name] = pd.read_csv(file_path, encoding='utf8') # Specify encoding
                print(f"Loaded {file} successfully ({len(dataframes[table_name])} rows)")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"Warning: {file} not found in data directory")

    return dataframes

# --- Data Processing and Feature Engineering ---

def parse_time_to_seconds(time_str):
    """
    Convert F1 time strings (like 'M:SS.ms' or 'SS.ms') to total seconds.
    Handles NaNs and potential formatting issues.
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    try:
        # Ensure the format has at least minutes and seconds for timedelta
        if ':' not in time_str:
             # Assume it's seconds.milliseconds
             time_str = '0:' + time_str
        # Prepend standard '0 days' for robust timedelta parsing
        return pd.to_timedelta('0 days 00:' + time_str).total_seconds()
    except (ValueError, TypeError):
        # Handle unexpected formats gracefully
        return None

def process_f1_data(f1_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Process and merge relevant F1 datasets for race prediction.
    Includes integration of sprint results and robust numeric conversion.
    """
    # Extract individual dataframes
    races = f1_data.get('races')
    results = f1_data.get('results')
    drivers = f1_data.get('drivers')
    constructors = f1_data.get('constructors')
    circuits = f1_data.get('circuits')
    qualifying = f1_data.get('qualifying')
    sprint_results = f1_data.get('sprint_results') # Get sprint results

    # Check if required dataframes are available
    required_dfs = {'races': races, 'results': results, 'drivers': drivers,
                    'constructors': constructors, 'circuits': circuits}
    if any(df is None for df in required_dfs.values()):
        missing = [name for name, df in required_dfs.items() if df is None]
        print(f"Error: Missing required dataframes: {missing}. Cannot proceed.")
        return None, None

    print("Processing and merging data...")

    # --- Clean column names and select relevant columns ---
    if 'url' in races.columns: races = races.rename(columns={'url': 'race_url'})
    if 'url' in circuits.columns: circuits = circuits.rename(columns={'url': 'circuit_url'})
    if 'url' in drivers.columns: drivers = drivers.rename(columns={'url': 'driver_url'})
    if 'url' in constructors.columns: constructors = constructors.rename(columns={'url': 'constructor_url'})
    if 'name' in circuits.columns: circuits = circuits.rename(columns={'name': 'circuit_name'})
    if 'name' in races.columns: races = races.rename(columns={'name': 'race_name'})
    if 'name' in constructors.columns: constructors = constructors.rename(columns={'name': 'constructor_name'})

    drivers = drivers[['driverId', 'driverRef', 'forename', 'surname', 'dob', 'nationality']]
    constructors = constructors[['constructorId', 'constructorRef', 'constructor_name', 'nationality']]
    races = races[['raceId', 'year', 'round', 'circuitId', 'race_name', 'date']]
    circuits = circuits[['circuitId', 'circuit_name', 'location', 'country']]
    results = results[['resultId', 'raceId', 'driverId', 'constructorId', 'number', 'grid', 'position', 'positionText', 'positionOrder', 'points', 'laps', 'time', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed', 'statusId']]
    # Convert results grid and positionOrder early
    results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
    results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')


    # --- Merge core data ---
    df = results.merge(drivers, on='driverId', how='left')
    df = df.merge(constructors, on='constructorId', how='left')
    df = df.merge(races, on='raceId', how='left')
    df = df.merge(circuits, on='circuitId', how='left')

    # --- Merge Qualifying Data ---
    if qualifying is not None:
        qual_data = qualifying[['raceId', 'driverId', 'position', 'q1', 'q2', 'q3']].copy()
        # Convert qualifying position to numeric early
        qual_data['position'] = pd.to_numeric(qual_data['position'], errors='coerce')
        qual_data.rename(columns={'position': 'qualifyingPosition'}, inplace=True)
        df = df.merge(qual_data, on=['raceId', 'driverId'], how='left')
        print("Merged Qualifying data.")

        for q_col in ['q1', 'q2', 'q3']:
            if q_col in df.columns:
                df[f'{q_col}_seconds'] = df[q_col].apply(parse_time_to_seconds)
                df.drop(columns=[q_col], inplace=True)
    else:
        print("Warning: Qualifying data not found or loaded.")
        df['qualifyingPosition'] = pd.NA

    # --- Merge Sprint Results Data ---
    if sprint_results is not None:
        sprint_data = sprint_results[['raceId', 'driverId', 'grid', 'position', 'points']].copy()
        # *** FIX: Convert sprint grid and position to numeric ***
        sprint_data['grid'] = pd.to_numeric(sprint_data['grid'], errors='coerce')
        sprint_data['position'] = pd.to_numeric(sprint_data['position'], errors='coerce')
        sprint_data['points'] = pd.to_numeric(sprint_data['points'], errors='coerce') # Also ensure points are numeric

        sprint_data.rename(columns={
            'grid': 'sprint_grid',
            'position': 'sprint_position',
            'points': 'sprint_points'
        }, inplace=True)
        df = df.merge(sprint_data, on=['raceId', 'driverId'], how='left')
        print("Merged Sprint Results data.")
    else:
        print("Warning: Sprint Results data not found or loaded.")
        df['sprint_grid'] = pd.NA
        df['sprint_position'] = pd.NA
        df['sprint_points'] = pd.NA

    # --- Feature Engineering ---
    print("Engineering features...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

    # Use the already numeric 'positionOrder' as the target 'position'
    df['position'] = df['positionOrder']

    # --- Performance Metrics ---
    df.sort_values(by=['date', 'round'], inplace=True)

    # Use expanding mean on the numeric 'position' column
    driver_avg_pos = df.groupby('driverId')['position'].transform(lambda x: x.expanding().mean().shift(1))
    constructor_avg_pos = df.groupby('constructorId')['position'].transform(lambda x: x.expanding().mean().shift(1))
    df['driver_avg_position'] = driver_avg_pos
    df['constructor_avg_position'] = constructor_avg_pos

    circuit_driver_avg_pos = df.groupby(['circuitId', 'driverId'])['position'].transform(lambda x: x.expanding().mean().shift(1))
    circuit_constructor_avg_pos = df.groupby(['circuitId', 'constructorId'])['position'].transform(lambda x: x.expanding().mean().shift(1))
    df['driver_circuit_avg_position'] = circuit_driver_avg_pos
    df['constructor_circuit_avg_position'] = circuit_constructor_avg_pos

    full_processed_data = df.copy()

    # --- Select Features for Modeling ---
    feature_cols = [
        'raceId', 'driverId', 'constructorId',
        'circuitId', 'grid', 'year', 'round',
        'driver_age',
        'driver_avg_position', 'constructor_avg_position',
        'driver_circuit_avg_position', 'constructor_circuit_avg_position',
        'qualifyingPosition', 'q1_seconds', 'q2_seconds', 'q3_seconds',
        'sprint_grid', 'sprint_position', 'sprint_points',
        'position', # Target variable
        'points'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    model_features = df[feature_cols].copy()

    # --- Handle Missing Values ---
    print("Handling missing values...")

    # Fill NaN target positions (should be fewer now with positionOrder)
    # Calculate max_pos *after* converting to numeric
    max_pos = model_features['position'].max() if not model_features['position'].isna().all() else 25
    model_features['position'].fillna(max_pos + 1, inplace=True)

    # Fill points
    if 'points' in model_features.columns:
         model_features['points'].fillna(0, inplace=True)
    if 'sprint_points' in model_features.columns:
        model_features['sprint_points'].fillna(0, inplace=True)

    # Fill grid positions (main, qual, sprint) - use median
    # Calculate medians *after* columns are confirmed numeric
    grid_median = model_features['grid'].median() if not model_features['grid'].isna().all() else 20
    model_features['grid'].fillna(grid_median, inplace=True)

    if 'qualifyingPosition' in model_features.columns:
        qual_pos_median = model_features['qualifyingPosition'].median() if not model_features['qualifyingPosition'].isna().all() else grid_median
        model_features['qualifyingPosition'].fillna(qual_pos_median, inplace=True)

    if 'sprint_grid' in model_features.columns:
        sprint_grid_median = model_features['sprint_grid'].median() if not model_features['sprint_grid'].isna().all() else grid_median
        model_features['sprint_grid'].fillna(sprint_grid_median, inplace=True)

    # Fill sprint position (use high number)
    # Calculate max_sprint_pos *after* column is confirmed numeric
    if 'sprint_position' in model_features.columns:
        # *** This line should now work correctly ***
        max_sprint_pos = model_features['sprint_position'].max() if not model_features['sprint_position'].isna().all() else 25
        model_features['sprint_position'].fillna(max_sprint_pos + 1, inplace=True)

    # Fill performance averages and other numeric columns
    numeric_cols_to_fill = model_features.select_dtypes(include=np.number).columns.drop('position', errors='ignore')
    for col in numeric_cols_to_fill:
        if model_features[col].isnull().any():
            mean_val = model_features[col].mean() # Or use median: model_features[col].median()
            model_features[col].fillna(mean_val, inplace=True)

    # Final check for any remaining NaNs (e.g., age if DOB missing)
    model_features.fillna(model_features.median(numeric_only=True), inplace=True)

    model_features.dropna(subset=['position'], inplace=True)
    full_processed_data = full_processed_data.loc[model_features.index]

    print(f"Processed data shape: {model_features.shape}")
    print(f"Final columns for modeling: {model_features.columns.tolist()}")

    model_features['position'] = model_features['position'].astype(int)

    return model_features, full_processed_data

# --- Model Training ---

def build_ml_models(processed_data: pd.DataFrame) -> tuple[dict, str]:
    """
    Build and evaluate machine learning models for race position prediction.
    Includes updated hyperparameters for potentially lower error.
    """
    print("Building and evaluating models...")
    if processed_data is None or processed_data.empty:
        print("Error: No processed data available for model training.")
        return {}, ""

    # Define features (X) and target (y)
    # Drop identifiers and the target variable itself, plus points (often considered a result, not a predictor)
    features_to_drop = ['position', 'points', 'raceId', 'driverId', 'constructorId', 'resultId', 'positionText', 'positionOrder', 'statusId', 'time', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed']
    X = processed_data.drop(columns=[col for col in features_to_drop if col in processed_data.columns], axis=1)
    y = processed_data['position']

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Features used: {X.columns.tolist()}")

    if X.empty:
        print("Error: No features remaining after dropping columns.")
        return {}, ""

    # Split data into training and testing sets (chronological split might be better for time series)
    # Using random split here for simplicity as in the original code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical and numerical columns *in the final feature set X*
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    # Create preprocessing pipeline
    # Handle cases where there are no numerical or categorical columns
    transformers = []
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    if cat_cols:
        # handle_unknown='ignore' is important for potential new categories in test set
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))

    if not transformers:
        print("Error: No numerical or categorical features identified for preprocessing.")
        return {}, ""

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' # Keep any columns not specified (shouldn't be any here)
    )

    # Define models to evaluate with adjusted hyperparameters
    # These are starting points; further tuning via GridSearchCV/RandomizedSearchCV is recommended.
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,  # Increased estimators
            random_state=42,
            n_jobs=-1,
            max_depth=20,      # Limit tree depth
            min_samples_split=5 # Require more samples to split
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,  # Increased estimators
            random_state=42,
            learning_rate=0.05, # Reduced learning rate
            max_depth=5,       # Limit tree depth
            subsample=0.8      # Use a fraction of samples for each tree
        ),
        'XGBoost': XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            n_estimators=200,  # Increased estimators
            learning_rate=0.05, # Reduced learning rate
            max_depth=6,       # Limit tree depth
            subsample=0.8,     # Use a fraction of samples for each tree
            colsample_bytree=0.8, # Use a fraction of features for each tree
            reg_alpha=0.1,     # L1 regularization
            reg_lambda=0.1     # L2 regularization
        )
    }

    results = {}
    best_model_name = ""
    best_model_metric = float('inf') # Looking for lowest MSE

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            results[name] = {'model': pipeline, 'mse': mse, 'mae': mae, 'r2': r2}

            # Track best model based on MSE
            if mse < best_model_metric:
                best_model_metric = mse
                best_model_name = name

        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            results[name] = {'model': None, 'mse': float('inf'), 'mae': float('inf'), 'r2': float('-inf')}


    if best_model_name:
        print(f"Best model based on MSE: {best_model_name} (MSE: {best_model_metric:.4f})")
    else:
        print("Warning: No model trained successfully.")

    return results, best_model_name

# --- Model Saving/Loading ---

def save_model(model_results: dict, best_model_name: str, models_dir: str) -> str | None:
    """Save the best model pipeline and performance metrics."""
    if not best_model_name or best_model_name not in model_results or model_results[best_model_name]['model'] is None:
        print("Error: Cannot save model. Best model not found or not trained successfully.")
        return None

    best_model = model_results[best_model_name]['model']
    model_path = os.path.join(models_dir, f'{best_model_name}_f1_model.joblib')
    try:
        joblib.dump(best_model, model_path)
        print(f"Saved {best_model_name} model pipeline to {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
        return None

    # Save model metrics
    metrics = {name: {'mse': res['mse'], 'mae': res['mae'], 'r2': res['r2']}
               for name, res in model_results.items()}
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = os.path.join(models_dir, 'f1_model_metrics.csv')
    try:
        metrics_df.to_csv(metrics_path)
        print(f"Saved model metrics to {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics to {metrics_path}: {e}")

    return model_path

def load_trained_model(model_path: str):
    """Load a trained model pipeline from file."""
    try:
        model = joblib.load(model_path)
        print(f"Loaded model pipeline from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# --- Prediction ---

def predict_race(model: Pipeline, race_id: int, processed_features_all: pd.DataFrame, full_data_all: pd.DataFrame) -> tuple[pd.DataFrame | None, str, int]:
    """
    Predict results for a specific race using the trained model pipeline.
    """
    if model is None:
        print("Error: Model not loaded.")
        return None, "N/A", 0

    # Get race details for display from the original full data
    race_info_display = full_data_all[full_data_all['raceId'] == race_id].copy()
    if race_info_display.empty:
        print(f"Error: No data found for raceId {race_id} in full data.")
        return None, "N/A", 0

    race_name = race_info_display['race_name'].iloc[0]
    race_year = race_info_display['year'].iloc[0]

    # Get the corresponding processed features for this race
    race_features_processed = processed_features_all[processed_features_all['raceId'] == race_id].copy()
    if race_features_processed.empty:
        print(f"Error: No data found for raceId {race_id} in processed features.")
        # Attempt to return display data without predictions
        race_info_display['Predicted Position (Raw)'] = np.nan
        race_info_display['Predicted Position (Rounded)'] = np.nan
        race_info_display['Prediction Error'] = np.nan
        return race_info_display[[
             'forename', 'surname', 'constructor_name', 'position',
             'Predicted Position (Raw)', 'Predicted Position (Rounded)', 'Prediction Error'
        ]], race_name, race_year


    # Prepare features for prediction: must match columns used during training
    # Get feature names from the preprocessor step of the pipeline
    try:
        # Handle potential differences in how features are stored (depends on sklearn version/ColumnTransformer)
        if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names_in_model = model.named_steps['preprocessor'].get_feature_names_out()
        else: # Fallback for older versions
             # This part is complex as it needs to reconstruct names after OneHotEncoding etc.
             # A simpler (but less robust) way is to assume the columns used in build_ml_models are correct.
             features_to_drop = ['position', 'points', 'raceId', 'driverId', 'constructorId', 'resultId', 'positionText', 'positionOrder', 'statusId', 'time', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed']
             prediction_feature_cols = [col for col in processed_features_all.columns if col not in features_to_drop and col in race_features_processed.columns]
             # We will rely on the pipeline's preprocessor to select the correct columns passed during fit.
             print("Warning: Could not automatically get feature names from pipeline. Assuming columns match training.")


        # Select only the columns needed for prediction (the ones X_train had)
        # The pipeline's preprocessor should handle selecting the right ones based on fit.
        X_predict = race_features_processed.drop(columns=['position', 'points', 'raceId', 'driverId', 'constructorId'], errors='ignore')

        # Ensure column order matches training (less critical with ColumnTransformer by name, but good practice)
        # X_predict = X_predict[training_columns] # Where training_columns is saved list from build_ml_models

        print(f"Predicting for Race ID: {race_id} ({race_year} {race_name}) using {len(X_predict)} entries.")
        print(f"Prediction features shape: {X_predict.shape}")
        print(f"Prediction feature columns: {X_predict.columns.tolist()}")

        # Make predictions
        predicted_positions = model.predict(X_predict)

        # Add predictions back to the display dataframe
        # Ensure index alignment if rows were dropped
        race_info_display['predicted_position_raw'] = predicted_positions
        race_info_display['predicted_position_rounded'] = np.round(predicted_positions).astype(int)

        # Ensure 'position' is numeric for error calculation
        race_info_display['position'] = pd.to_numeric(race_info_display['position'], errors='coerce')
        race_info_display['predicted_position_rounded'] = pd.to_numeric(race_info_display['predicted_position_rounded'], errors='coerce')

        # Calculate prediction error (handle potential NaNs)
        race_info_display['prediction_error'] = race_info_display['position'] - race_info_display['predicted_position_rounded']

    except Exception as e:
        print(f"Error during prediction for race {race_id}: {e}")
        # Add empty prediction columns if prediction fails
        race_info_display['predicted_position_raw'] = np.nan
        race_info_display['predicted_position_rounded'] = np.nan
        race_info_display['prediction_error'] = np.nan

    # Prepare final results table
    results_table = race_info_display.sort_values('position')[[
        'forename', 'surname', 'constructor_name', 'grid', 'position',
        'predicted_position_raw', 'predicted_position_rounded', 'prediction_error'
    ]].copy()

    results_table.rename(columns={
        'forename': 'Driver First Name',
        'surname': 'Driver Last Name',
        'constructor_name': 'Team',
        'grid': 'Grid',
        'position': 'Actual Position',
        'predicted_position_raw': 'Predicted Position (Raw)',
        'predicted_position_rounded': 'Predicted Position (Rounded)',
        'prediction_error': 'Prediction Error'
    }, inplace=True)

    return results_table, race_name, race_year


# --- Gradio UI ---

def train_and_get_data():
    """Train models and return the path to the best model and the full dataset."""
    print("--- Starting Model Training ---")
    f1_data = load_f1_data(data_dir)
    if not f1_data:
        print("Critical Error: Could not load base data. Exiting.")
        return None, None, None

    processed_data, full_data = process_f1_data(f1_data)
    if processed_data is None or full_data is None:
        print("Critical Error: Data processing failed. Exiting.")
        return None, None, None

    results, best_model_name = build_ml_models(processed_data)
    if not best_model_name:
         print("Critical Error: Model training failed. Exiting.")
         return None, None, None # Return None if training failed

    model_path = save_model(results, best_model_name, models_dir)
    if model_path is None:
        print("Critical Error: Failed to save the model. Exiting.")
        return None, None, None

    print("--- Model Training Complete ---")
    return model_path, processed_data, full_data


def get_available_races(full_data: pd.DataFrame) -> dict[str, int]:
    """Get list of available races for the UI dropdown."""
    if full_data is None or not {'raceId', 'year', 'race_name'} .issubset(full_data.columns):
        return {"No races available": -1}

    # Get unique races with year and name
    races = full_data[['raceId', 'year', 'race_name']].drop_duplicates().copy()

    # Ensure year is integer for sorting
    races['year'] = races['year'].astype(int)

    # Sort by year (descending) and race name (ascending)
    races = races.sort_values(['year', 'race_name'], ascending=[False, True])

    # Create race options for dropdown: "YYYY Race Name" -> raceId
    race_options = {
        f"{int(row['year'])} {row['race_name']}": int(row['raceId'])
        for _, row in races.iterrows()
    }

    return race_options

def create_gradio_interface(model_path: str | None, processed_data: pd.DataFrame | None, full_data: pd.DataFrame | None):
    """Create Gradio interface for race prediction."""

    if model_path is None or processed_data is None or full_data is None:
         print("Error: Cannot create Gradio interface due to missing model or data.")
         # Create a dummy interface showing the error
         with gr.Blocks() as iface:
              gr.Markdown("# F1 Prediction System Error")
              gr.Markdown("Failed to load model or process data. Please check the console logs.")
         return iface

    # Load the trained model
    model = load_trained_model(model_path)
    if model is None:
         print("Error: Failed to load the trained model.")
         with gr.Blocks() as iface:
              gr.Markdown("# F1 Prediction System Error")
              gr.Markdown("Failed to load the trained model file. Please check the console logs.")
         return iface

    # Get list of races for the dropdown
    race_options = get_available_races(full_data)
    if not race_options or list(race_options.keys())[0] == "No races available":
         print("Error: No races available to display.")
         with gr.Blocks() as iface:
              gr.Markdown("# F1 Prediction System Error")
              gr.Markdown("No race data found or processed correctly.")
         return iface

    # Define the Gradio interface function
    def predict_and_display(race_selection: str) -> tuple[str, str, pd.DataFrame | None]:
        if race_selection is None or race_selection not in race_options:
             return "## Select a Race", "Please choose a race from the dropdown menu.", None

        race_id = race_options[race_selection]
        results_table, race_name, race_year = predict_race(model, race_id, processed_data, full_data)

        if results_table is None:
            return f"## Error Predicting Race", f"Could not generate predictions for {race_year} {race_name}.", None

        # Create title
        title = f"## {race_year} {race_name} - Race Results vs Predictions"

        # Format the results table for display
        try:
            # Get the winner (actual position == 1)
            winner_row = results_table[results_table['Actual Position'] == 1]
            if not winner_row.empty:
                 winner = winner_row.iloc[0]
                 winner_name = f"{winner['Driver First Name']} {winner['Driver Last Name']}"
                 winner_team = winner['Team']
                 winner_info = f"### Race Winner: {winner_name} ({winner_team})"
            else:
                 winner_info = "### Race Winner: Not Found (Position 1 missing)"

            # Prediction accuracy stats (handle NaNs in error calculation)
            valid_errors = results_table['Prediction Error'].dropna()
            if not valid_errors.empty:
                mean_abs_error = abs(valid_errors).mean()
                exact_predictions = (valid_errors == 0).sum()
                total_drivers = len(results_table) # Count all drivers displayed
                accuracy_pct = (exact_predictions / len(valid_errors)) * 100 # Accuracy based on valid predictions
                perf_summary = f"""
**Prediction Performance:**
- Mean Absolute Error (MAE): {mean_abs_error:.2f} positions (on {len(valid_errors)} drivers with valid actual/predicted positions)
- Exact Position Predictions: {exact_predictions} out of {len(valid_errors)} ({accuracy_pct:.1f}%)
                """
            else:
                perf_summary = "**Prediction Performance:**\n- No valid prediction errors to calculate metrics."

            summary = f"{winner_info}\n\n{perf_summary}"

        except Exception as e:
            print(f"Error generating summary for race {race_id}: {e}")
            summary = "Error generating summary information."

        return title, summary, results_table

    # Create the interface
    iface = gr.Interface(
        fn=predict_and_display,
        inputs=[
            gr.Dropdown(
                choices=list(race_options.keys()),
                label="Select Race",
                value=list(race_options.keys())[0] if race_options else None # Default to most recent race
            )
        ],
        outputs=[
            gr.Markdown(label="Race Title"),
            gr.Markdown(label="Summary"),
            gr.DataFrame(label="Race Results Comparison", wrap=True) # Allow wrapping for smaller screens
        ],
        title="F1 Race Position Prediction System",
        description="Select a past race to see the model's predicted finishing positions compared to the actual results. Lower 'Actual Position' is better.",
        theme="default" # Or try alternatives like "huggingface", "soft", "glass"
    )

    return iface

# --- Main Execution ---

def main():
    """Main function to run the F1 prediction system."""
    print("--- F1 Race Prediction System ---")
    print(f"Data Directory: {data_dir}")
    print(f"Models Directory: {models_dir}")

    # --- Check for existing model or train ---
    model_path = None
    processed_data = None
    full_data = None

    # Look for a specific model file pattern
    model_pattern = "_f1_model.joblib"
    existing_models = [f for f in os.listdir(models_dir) if f.endswith(model_pattern)] if os.path.exists(models_dir) else []

    if existing_models:
        model_path = os.path.join(models_dir, existing_models[0]) # Load the first one found
        print(f"Found existing model: {model_path}. Loading model and data...")
        # Need to load data even if model exists, for the UI selection and display
        f1_data = load_f1_data(data_dir)
        if f1_data:
             processed_data, full_data = process_f1_data(f1_data)
             if processed_data is None or full_data is None:
                 print("Error processing data even though model exists. Retraining might be needed.")
                 model_path = None # Force retraining if data processing fails
             else:
                 print("Data loaded successfully.")
        else:
             print("Error loading data even though model exists. Retraining might be needed.")
             model_path = None # Force retraining if data loading fails

    # If no model found, or if loading/processing data failed with existing model
    if model_path is None:
        print("No suitable existing model found or data loading failed. Training new model...")
        model_path, processed_data, full_data = train_and_get_data()
        if model_path is None:
             print("Exiting due to training failure.")
             return # Exit if training fails

    # --- Create and launch Gradio interface ---
    if model_path and processed_data is not None and full_data is not None:
        print("Creating Gradio interface...")
        iface = create_gradio_interface(model_path, processed_data, full_data)

        print("Launching Gradio interface. Access it in your web browser (check console for URL).")
        iface.launch()
    else:
        print("Could not launch Gradio interface due to missing model or data after attempting load/train.")

if __name__ == "__main__":
    main()
