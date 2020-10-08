import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
def _preprocess_data():
    '''
    This component will
    1. Load the presidential elections datasets
    2. Clean and transoform the datasets
    3. Split the dataset into train and test sets
    4. Use np.save to save our dataset to disk so that it can be reused by later components
    '''
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    np.save('x_train.npy', X_train)
    np.save('x_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    print('Preprocessing presidential elections data...')
    _preprocess_data()
