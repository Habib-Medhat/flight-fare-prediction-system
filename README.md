# Flight Fare EDA and ML Project

This project explores Bangladesh flight fare data and exposes a FastAPI endpoint for fare prediction.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API

```powershell
uvicorn flight_fare_api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Run the EDA script

```powershell
python eda_analysis.py
```

The script loads `bangladesh_flight_prices_dataset.csv`, prepares date/time features, caps fare outliers, runs statistical tests, applies PCA, and compares regression models.

## Frontend

Open `flight_fare_frontend.html` after starting the API. The form uses the same feature names and category values expected by the model.
