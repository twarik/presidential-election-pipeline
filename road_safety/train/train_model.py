import argparse
import joblib
import numpy as np

from fbprophet import Prophet

def predict(train):

    train_data = np.load(train, allow_pickle=True)

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True,seasonality_mode='multiplicative',
                daily_seasonality=True)

    model.fit(train_data)

    #Save the model
    joblib.dump(model, 'model.pkl')

    print("Model Trained")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    args = parser.parse_args()
    train_model(args.train)
