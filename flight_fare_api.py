from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from flight_utils import build_feature_matrix, make_prediction_frame, train_fare_model


class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Stopovers: str
    Aircraft_Type: str
    Class: str
    Booking_Source: str
    Seasonality: str
    Departure_Day: int = Field(ge=1, le=31)
    Departure_Month: int = Field(ge=1, le=12)
    Departure_Hour: int = Field(ge=0, le=23)
    Departure_Weekday: int = Field(ge=0, le=6)
    Base_Fare: float = Field(gt=0)
    Tax_Surcharge: float = Field(ge=0)


model = None
encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoder
    model, encoder, _ = train_fare_model()
    yield


app = FastAPI(title="Flight Fare Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Flight fare API is running"}


@app.post("/predict")
def predict_fare(data: FlightInput):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    input_df = make_prediction_frame(data)
    X, _, _ = build_feature_matrix(input_df, encoder)
    prediction = model.predict(X)

    return {"predicted_total_fare": round(float(prediction[0]), 2)}


@app.get("/predict/example")
def sample_prediction():
    example_data = FlightInput(
        Airline="Biman Bangladesh",
        Source="DAC",
        Destination="CGP",
        Stopovers="Direct",
        Aircraft_Type="Boeing 737",
        Class="Economy",
        Booking_Source="Online Website",
        Seasonality="Regular",
        Departure_Day=12,
        Departure_Month=5,
        Departure_Hour=9,
        Departure_Weekday=2,
        Base_Fare=5500.0,
        Tax_Surcharge=1300.0,
    )
    return predict_fare(example_data)
