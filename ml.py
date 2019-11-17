import os
import pandas as pd


class MlProblem:
    def __init__(self, name, path_train="", path_test="", id_col="", cible_col=""):
        if os.path.exists(path_train) & os.path.exists(path_test):
            self.name = name
            self.train_set_path = path_train
            self.test_set_path = path_test
            self.df_train = self.get_train_set_from_path()
            self.df_test = self.get_test_set_from_path()
            self.all_col_train = self.df_train.columns.tolist()
            self.all_col_test = self.df_test.columns.tolist()
            self.cible_col = cible_col
            self.id_col = id_col
        else:
            raise ValueError("Path to train or test or both doesn't exist")

    def set_path_to_train_set(self, path):
        self.train_set_path = path

    def get_path_to_train_set(self):
        return self.train_set_path

    def set_path_to_test_set(self, path):
        self.test_set_path = path

    def get_path_to_test_set(self):
        return self.test_set_path

    def get_train_set_from_path(self, delim=None, engine="python", encoding="utf8"):
        """Create the df_train

        :param delim:
        :param engine:
        :param encoding:
        :return:
        """
        path_train = self.get_path_to_train_set()
        assert(path_train.split(".")[-1] == "csv"), "Must be a csv"
        return pd.read_csv(path_train, sep=delim, engine=engine, verbose=True, encoding=encoding)

    def get_test_set_from_path(self, delim=None, engine="python", encoding="utf8"):
        """

        :param delim:
        :param engine:
        :param encoding:
        :return:
        """
        path_test = self.get_path_to_test_set()
        assert(path_test.split(".")[-1] == "csv"), "Must be a csv"
        return pd.read_csv(path_test, sep=delim, engine=engine, verbose=True, encoding=encoding)

    def set_cible_col(self, cible_col):
        self.cible_col = cible_col

    def set_id_col(self, id_col):
        self.id_col = id_col

    def get_X_train_raw(self):
        return self.df_train[[col for col in self.all_col_train if col not in self.id_col + self.cible_col]]

    def get_y_train_raw(self):
        return self.df_train[self.cible_col].values

    def get_X_to_predict_raw(self):
        return self.df_test[[col for col in self.all_col_test if col not in self.id_col]]


class Regression(MlProblem):
    type_ml = "r"

    def __init__(self, name, path_train, path_test, id_col, cible_col):
        MlProblem.__init__(self, name, path_train, path_test, id_col, cible_col)


class Classification(MlProblem):
    type_ml = "clf"

    def __init__(self, name, path_train, path_test, id_col, cible_col):
        MlProblem.__init__(self, name, path_train, path_test, id_col, cible_col)
