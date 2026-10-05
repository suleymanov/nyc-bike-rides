import os

import pandas as pd
from lightgbm import Booster

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


model_file = 'simple_models' + os.sep + 'model_lightgbm'
model = Booster(model_file=model_file)
app = FastAPI(
    title="Model service",
    description="Simple server for bike share prediction",
    version="0.1.0"
)


class SFeatures(BaseModel):
    end_station_name: str
    count_30: int
    count_60: int
    count_120: int
    month: int
    weekday: int
    hour: int
    hour_part: int


def _predict(end_station_name, count_30, count_60, count_120, month, weekday, hour, hour_part):
    X = pd.DataFrame([{
        'end_station_name': end_station_name,
        'count_30': count_30,
        'count_60': count_60,
        'count_120': count_120,
        'month': month,
        'weekday': weekday,
        'hour': hour,
        'hour_part': hour_part
    }])
    X['end_station_name'] = X['end_station_name'].astype('category')
    
    pred = model.predict(X)
    pred = int(round(pred[0]))
    pred = max(pred, 0)

    return pred


@app.post('/predict')
async def predict(features: SFeatures):
    features_dict = features.model_dump()
    pred = _predict(**features_dict)

    return pred


if __name__ == '__main__':
    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)
