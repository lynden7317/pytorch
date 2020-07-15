import os
import errno
import numpy as np
import xgboost as xgb

def svm_classifier():
    pass

def xgboost_classifier(num_classes=2,
                       x_data_path=None,
                       y_data_path=None,
                       steps=500,
                       param=None,
                       model_path=None,
                       save_path="./xgb_weights/xgb",
                       is_train=False):
    """
    :param num_classes:
    :param x_data_path: path to npy data format
    :param y_data_path: path to npy data format
    :param steps:
    :param param:
    :param model_path:
    :param save_path:
    :param is_train:
    :return:
    example:
        xgboost_classifier(x_data_path="../x_features.npy", y_data_path="../y_label.npy", is_train=True)
        xgboost_classifier(model_path="../xgb.model", is_train=False)
    """
    if param is None:
        param = {'eta': 0.1, 'max_depth': 5, 'objective': 'multi:softprob'}

    param["num_class"] = num_classes
    if is_train:
        if x_data_path is None or y_data_path is None:
            print("No data is provided during training")
            return None
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        X_train = np.load(x_data_path)
        Y_label = np.load(y_data_path)
        D_train = xgb.DMatrix(X_train, label=Y_label)
        watchlist = [(D_train, 'train')]
        xgb_model = xgb.train(param, D_train, steps, watchlist)
        save_name = os.path.basename(save_path)+str(steps)+'.model'
        xgb_model.save_model(os.path.join(save_dir, save_name))
    else:
        if model_path is None:
            print("No XGBoost pre-trained model")
            return None

        XGB = xgb.Booster(param)
        XGB.load_model(model_path)
        return (XGB, xgb.DMatrix, 'xgboost')

if __name__ == '__main__':
    xgboost_classifier(x_data_path="../x_features.npy", y_data_path="../y_label.npy", is_train=True)