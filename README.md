# Formula 1 Race Position Predictor 🏎️

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/%F0%9F%A4%97-Gradio-orange)](https://gradio.app)

An advanced machine learning project that predicts the finishing positions of drivers in Formula 1 races. This repository contains the code for data processing, model training with hyperparameter tuning, and an interactive web UI to visualize and explore predictions.

<!-- You can add a GIF or screenshot of the Gradio interface here! -->
<!-- ![Gradio UI Demo](link_to_your_gif_or_screenshot.png) -->

## ✨ Key Features

- **Accurate Predictions**: Utilizes a tuned `GradientBoostingRegressor` model to forecast race outcomes.
- **Rich Feature Engineering**: Incorporates a wide range of features, including:
    - Historical driver and constructor performance (average finishing positions).
    - Track-specific performance averages.
    - Qualifying results and lap times (`q1`, `q2`, `q3`).
    - Sprint race outcomes (grid, position, points).
    - Driver-specific attributes like age.
    - Polynomial features for grid position to capture non-linear trends.
- **Interactive Web UI**: A user-friendly interface built with **Gradio** to select any past race and see the model's predictions compared against the actual results.
- **Performance Analytics**: Automatically calculates and displays Mean Absolute Error (MAE) and R² scores for each prediction, providing immediate insight into the model's accuracy.
- **Chronological Data Handling**: Employs a time-based train-test split to ensure the model is evaluated on future data it has never seen, simulating a real-world prediction scenario.
- **Automated Pipeline**: The entire process, from data loading and processing to model training and saving, is automated in a single script.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Scikit-learn**: For machine learning pipelines, preprocessing, and model evaluation.
- **XGBoost**: Used as one of the powerful gradient boosting models.
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical operations.
- **Gradio**: To create the interactive web application.
- **Joblib**: For saving and loading the trained model pipeline.

## 📊 Dataset

This project is powered by the comprehensive [Ergast F1 API](http://ergast.com/mrd/), using the following datasets stored in the `/data` directory:
- `races.csv`
- `results.csv`
- `qualifying.csv`
- `sprint_results.csv`
- `drivers.csv`
- `constructors.csv`
- `circuits.csv`
- and other supporting files.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Prathmesh333/F1-Race-Prediction.git
    cd F1-Race-Prediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To start the application, run the main script from the project root directory:

```bash
python f1_final.py
```

- The first time you run the script, it will:
    1.  Load and process all the data from the `/data` directory.
    2.  Train multiple regression models.
    3.  Perform hyperparameter tuning for the `GradientBoostingRegressor` to find the best parameters.
    4.  Save the best-performing model pipeline to the `/models` directory.
- On subsequent runs, the script will automatically load the saved model, skipping the lengthy training process.
- Once the model is ready, the script will launch a Gradio web server. You can access the interactive UI by navigating to the local URL provided in your terminal (usually `http://127.0.0.1:7860`).

## 🤖 Model & Methodology

The core of this project is a `GradientBoostingRegressor` model. The methodology is as follows:

1.  **Data Loading**: All relevant CSV files are loaded into Pandas DataFrames.
2.  **Data Processing**: The datasets are merged into a single, comprehensive DataFrame.
3.  **Feature Engineering**: New features are created to provide the model with deeper context, as listed in the features section.
4.  **Chronological Split**: The data is split into a training set (races before a certain point in time) and a testing set (races after that point). This prevents data leakage and ensures the model is tested on unseen future data.
5.  **Preprocessing**: A `ColumnTransformer` pipeline is used to apply standard scaling to numerical features and one-hot encoding to categorical features.
6.  **Hyperparameter Tuning**: `RandomizedSearchCV` is used to efficiently search for the optimal hyperparameters for the `GradientBoostingRegressor`, optimizing for the lowest Mean Squared Error.
7.  **Training & Evaluation**: The final model is trained on the full training set and evaluated on the test set to measure its real-world performance.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
