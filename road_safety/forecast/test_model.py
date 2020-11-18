import argparse
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from fbprophet import Prophet

def test_model(test, model_path):
    test_data = np.load(x_test, allow_pickle=True)

    model = joblib.load(model_path)

    #Evaluate model and print results
    future_date = model.make_future_dataframe(periods=150, freq='D')

    forecast = model.predict(future_date)

    print('Model \nr2_score = {} \nMAE = {} \nMSE = {}' .format(r2_score(forecast.trend, forecast.yhat),
    MAE (forecast.trend, forecast.yhat), MSE (forecast.trend, forecast.yhat) ))

    #save result
    with open(f'model_result.txt', 'w') as result:
        result.write("Prediction:{},\nActual: {}".format(forecast, test))


    print("Prediction saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test')
    parser.add_argument('--model')
    args = parser.parse_args()
    test_model(args.x_test, args.model)
