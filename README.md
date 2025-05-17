# Cryptocurrency Liquidity Predictor

## Problem Statement

In this project, you will build a machine learning system to predict the liquidity of cryptocurrencies using historical market and price data. This project is useful for traders, exchanges, and analysts who want to assess the ease of buying or selling a cryptocurrency without affecting its price.

## Solution Proposed

The goal is to dynamically predict the liquidity of a cryptocurrency based on its market and trading features. We use a machine learning approach, where we engineer relevant features, train regression models, and deploy the solution for real-time predictions.

### Dataset Used
- Dataset link:https://drive.google.com/drive/folders/10BRgPip2Zj_56is3DilJCowjfyT6E9AM

## Tech Stack Used

1. Python
2. Streamlit
3. Scikit-learn
4. Pandas, NumPy, Matplotlib, Seaborn

## Infrastructure Required

1. Local machine or cloud VM
2. (Optional) GitHub Actions for CI/CD

## How to Run

Step 1. Clone the repository.

```
git clone <your-repo-url>
cd Crytocurrency\ Liquidity\ predictor
```

Step 2. (Optional) Create a virtual environment.

```
python -m venv venv
.\venv\Scripts\activate
```

Step 3. Install the requirements

```
pip install -r requirements.txt
```

Step 4. Train the model

```
python src/model.py
```

Step 5. Run the Streamlit app

```
streamlit run app.py
```

## Project Architecture

- **Data Layer:** CSV files in `notebook/`
- **Processing Layer:** Scripts in `src/` for preprocessing, feature engineering, and modeling
- **Model Layer:** Trained model and scaler in `models/`
- **Interface Layer:** Streamlit app in `app.py`

## Notebooks
- `EDA.ipynb`: Exploratory Data Analysis
- `Feature_Engineering.ipynb`: Feature creation and transformation
- `Feature_Selection.ipynb`: Feature selection and model evaluation

## Models Used

* RandomForestRegressor (with GridSearchCV for hyperparameter optimization)
* Feature scaling with StandardScaler

## src Folder Structure

- `feature.py`: Feature engineering functions
- `model.py`: Model training and saving
- `preprocess.py`: Data loading and cleaning
- `utils.py`: Utility functions for plotting and analysis

## Conclusion

- This project demonstrates a complete ML pipeline for predicting cryptocurrency liquidity, from data ingestion to deployment.
- The codebase is modular, well-documented, and ready for further extension.

---

> For more details, see the notebooks and code comments.
