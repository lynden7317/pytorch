import numpy as np
import xgboost as xgb

def xgboost_classifier():
    steps = 500  # The number of training iterations
    saveName = 'resnet50_xgb_'
    x_path = '../x_features.npy'
    y_path = '../y_label.npy'
    num_classes = 2

    param = {
        'eta': 0.1,
        'max_depth': 5,
        'objective': 'multi:softprob',
        'num_class': num_classes
    }

    X_train = np.load(x_path)
    Y_label = np.load(y_path)

    D_train = xgb.DMatrix(X_train, label=Y_label)
    watchlist = [(D_train, 'train')]

    xgb_model = xgb.train(param, D_train, steps, watchlist)

    saveName = saveName + str(steps) + '.model'
    xgb_model.save_model(saveName)

xgboost_classifier()