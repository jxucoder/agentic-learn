import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path('/Users/jiaruixu/work_space/agentic-learn/data/california_housing.csv')
RESULT_PATH = Path('result.json')
TARGET = 'MedHouseVal'
RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    eps = 1e-6

    # Ratio features
    x['BedroomsPerRoom'] = x['AveBedrms'] / (x['AveRooms'] + eps)
    x['RoomsPerPerson'] = x['AveRooms'] / (x['AveOccup'] + eps)
    x['PopPerHousehold'] = x['Population'] / (x['AveOccup'] + eps)
    x['IncomePerPerson'] = x['MedInc'] / (x['AveOccup'] + eps)

    # Income interaction features
    x['IncomeXRooms'] = x['MedInc'] * x['AveRooms']
    x['IncomeXAge'] = x['MedInc'] * x['HouseAge']

    # Spatial features
    x['LatLongInteraction'] = x['Latitude'] * x['Longitude']
    x['DistToSF'] = np.sqrt((x['Latitude'] - 37.7749) ** 2 + (x['Longitude'] + 122.4194) ** 2)
    x['DistToLA'] = np.sqrt((x['Latitude'] - 34.0522) ** 2 + (x['Longitude'] + 118.2437) ** 2)

    # Spatial clustering features
    coords = x[['Latitude', 'Longitude']]
    for k in (12, 24):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(coords)
        x[f'GeoCluster{k}'] = labels
        x[f'GeoDist{k}'] = km.transform(coords).min(axis=1)

    return x


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    y = df[TARGET]
    x = add_features(df.drop(columns=[TARGET]))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = ExtraTreesRegressor(
        n_estimators=1000,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_features='sqrt',
        min_samples_leaf=1,
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metric = float(r2_score(y_test, preds))

    payload = {'metric': metric}
    RESULT_PATH.write_text(json.dumps(payload), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
