📈 Sales Forecasting System

📌 About the Project

The Sales Forecasting System is a Machine Learning-based application designed to analyze historical sales data and predict future sales. The project performs data preprocessing, exploratory data analysis, feature engineering, model training, model comparison, and future sales forecasting.

The system uses Linear Regression and Random Forest Regression models to learn patterns from historical sales data and generate accurate sales predictions.

🎯 Objectives

📊 Analyze historical sales data
🧹 Perform data preprocessing and cleaning
🔍 Conduct Exploratory Data Analysis (EDA)
⚙️ Perform feature engineering
🤖 Train Machine Learning models
📈 Compare Linear Regression and Random Forest models
🎯 Select the best-performing model
🔮 Forecast future sales
📋 Generate predictions, metrics, graphs, and reports
🌐 Provide an interactive web interface using Flask

🛠️ Technologies Used

🐍 Python
🐼 Pandas
🔢 NumPy
🤖 Scikit-learn
📊 Matplotlib
📉 Seaborn
🌐 Flask
💾 Joblib
HTML
CSS
Git & GitHub

🤖 Machine Learning Models

The project includes:

Linear Regression
Random Forest Regression
Baseline Model
Model Comparison
Hyperparameter Tuning
Feature Importance Analysis

The trained models are stored in the models/ folder.

Project Workflow

Historical Sales Data
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train/Test Data Split
        ↓
Model Preparation
        ↓
Linear Regression
        ↓
Random Forest
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Sales Prediction
        ↓
Future Sales Forecasting
        ↓
Graphs, Metrics & Reports
        ↓
Flask Web Application

📂 Project Structure

Sales_Forecasting_System/
│
├── data/
│   ├── raw/
│   │   └── sales_data.csv
│   └── processed/
│       ├── processed_sales.csv
│       ├── sales_features.csv
│       ├── train_data.csv
│       └── test_data.csv
│
├── models/
│   ├── best_random_forest_model.pkl
│   ├── linear_regression_model.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│
├── outputs/
│   ├── graphs/
│   ├── metrics/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── baseline_model.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── feature_importance.py
│   ├── final_prediction.py
│   ├── future_forecasting.py
│   ├── hyperparameter_tuning.py
│   ├── interactive_forecasting.py
│   ├── linear_regression_model.py
│   ├── model_comparison.py
│   ├── model_preparation.py
│   └── random_forest_model.py
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── generate_dataset.py
├── README.md
├── requirements.txt
└── .gitignore

📊 Outputs

The project generates:

📈 Sales analysis graphs
📊 Model performance metrics
🔮 Sales predictions
📋 Forecasting reports
⭐ Feature importance results
📉 Model comparison results

🌐 Flask Web Application

The project includes a Flask-based web application that provides an interactive interface for generating sales predictions and viewing forecasting results.

⚙️ Installation

Clone the repository:

git clone <your-github-repository-url>

Open the project folder:

cd Sales_Forecasting_System

Create and activate a virtual environment:

py -m venv venv
venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt
▶️ Run the Application
python app.py

Then open:

http://127.0.0.1:5000

📌 Key Features

✅ Historical sales data processing
✅ Automated feature engineering
✅ Exploratory data analysis
✅ Multiple regression models
✅ Random Forest model
✅ Model comparison
✅ Hyperparameter tuning
✅ Feature importance analysis
✅ Future sales forecasting
✅ Interactive Flask interface
✅ Prediction and reporting

👩‍💻 Author

Anuradha

🎓 B.Tech – Artificial Intelligence & Data Science

Built with: Python, Pandas, NumPy, Scikit-learn, Random Forest, Linear Regression, Matplotlib, Seaborn, Flask, Joblib, HTML, CSS, Jupyter Notebook, Git, and GitHub.
