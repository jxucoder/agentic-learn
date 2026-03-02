import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path('/Users/jiaruixu/work_space/agentic-learn/data/california_housing.csv')
RESULT_PATH = Path('result.json')
TARGET = 'MedHouseVal'
RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    eps = 1e-6

    x['BedroomsPerRoom'] = x['AveBedrms'] / (x['AveRooms'] + eps)
    x['RoomsPerPerson'] = x['AveRooms'] / (x['AveOccup'] + eps)
    x['PopPerHousehold'] = x['Population'] / (x['AveOccup'] + eps)

    x['IncomeXRooms'] = x['MedInc'] * x['AveRooms']
    x['IncomeXOccup'] = x['MedInc'] * x['AveOccup']
    x['IncomeXAge'] = x['MedInc'] * x['HouseAge']

    x['LatLongInteraction'] = x['Latitude'] * x['Longitude']
    x['DistToSF'] = np.sqrt((x['Latitude'] - 37.7749) ** 2 + (x['Longitude'] + 122.4194) ** 2)
    x['DistToLA'] = np.sqrt((x['Latitude'] - 34.0522) ** 2 + (x['Longitude'] + 118.2437) ** 2)

    return x


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    y = df[TARGET]
    x = add_features(df.drop(columns=[TARGET]))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        learning_rate=0.05,
        max_depth=10,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_iter=800,
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metric = float(r2_score(y_test, preds))

    payload = {'metric': metric}
    RESULT_PATH.write_text(json.dumps(payload), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
