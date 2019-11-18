import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from utils import LabelEncodingColumns, CustomImputer, CustomOneHotEncoder


class Model:
    """
    Model class
    """
    def __init__(self, type_ml, X_train, y_train, X_test):
        self.type_ml = type_ml
        assert self.type_ml in ["clf", "r"], "Must be Classifier 'clf' or Regressor 'r' !"
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.n_features = X_train.shape[1]

    def encode_cible_if_classifier(self):
        if self.type_ml == "clf":
            le = LabelEncoder()
            cible_encoded = le.fit_transform(self.y_train)
            self.y_train = cible_encoded
            self.label_encode = le

    def get_model_and_params_for_ml_problem(self):
        if self.type_ml == "r":
            xg_r = XGBRegressor()
            rf_r = RandomForestRegressor()
            enet_r = ElasticNet()

            param_grid_r_xg = dict([("r__n_estimators", [150, 300]), ("r__learning_rate", np.linspace(0.05, 0.15, 4)),
                                    ("r__max_depth", [3, self.n_features // 2, self.n_features]), ("r__n_jobs", [-1]),
                                    ("r__silent ", [False])])
            param_grid_r_rf = dict([("r__n_estimators", [150, 300]), ("r__max_features", ["sqrt", "auto"]),
                                    ("r__max_depth", [3, self.n_features // 2, None]), ("r__n_jobs", [-1]),
                                    ("r__min_samples_leaf", [2, 4, 10]), ("r__verbose", [1])])
            param_grid_r_enet = dict([("r__alpha", [0, 1]), ("r___l1_ratio", [0, 0.5, 1]), ("r__normalize", [False])])
            return [xg_r, rf_r, enet_r], ["xg", "rf", "enet"], [param_grid_r_xg, param_grid_r_rf, param_grid_r_enet]
        else:
            xg_clf = XGBClassifier()
            rf_clf = RandomForestClassifier()
            log_clf = LogisticRegression()

            param_grid_clf_xg = dict([("clf__n_estimators", [150, 300]),
                                      ("clf__learning_rate", np.linspace(0.05, 0.15, 4)),
                                      ("clf__max_depth", [3, self.n_features // 2, self.n_features]), ("clf__n_jobs", [-1]),
                                      ("clf__silent", [False])])
            param_grid_clf_rf = dict([("clf__n_estimators", [150, 300]), ("clf__max_features", ["sqrt", "auto"]),
                                      ("clf__max_depth", [3, self.n_features // 2, None]), ("clf__n_jobs", [-1]),
                                      ("clf__min_samples_leaf", [2, 4, 10]), ("clf__class_weight", ["balanced"]),
                                      ("clf__verbose", [1])])
            param_grid_clf_log = dict([("clf__C", [0.001, 0.01, 0.1, 1, 10, 100]),
                                       ("clf__class_weight", ["balanced"]), ("clf__n_jobs", [-1])])

            return [xg_clf, rf_clf, log_clf], ["xg", "rf", "log"], [param_grid_clf_xg, param_grid_clf_rf, param_grid_clf_log]

    def set_model_and_params_for_ml(self):
        self.models_list, self.models_names_list, self.grids_list = self.get_model_and_params_for_ml_problem()

    def get_model_config_df(self):
        dict_tmp = {}
        model_kind = {"rf": "tree",
                      "xg": "tree",
                      "log": "linear",
                      "enet": "linear"}
        for i in range(len(self.models_list)):
            _name = self.models_names_list[i]
            dict_tmp[_name] = {}
            dict_tmp[_name]["_models"] = self.models_list[i]
            dict_tmp[_name]["_grids"] = self.grids_list[i]
            dict_tmp[_name]["_pipeSteps"] = (self.type_ml, dict_tmp[_name]["_models"])
            dict_tmp[_name]["_kind"] = model_kind[_name]
        return pd.DataFrame(dict_tmp)

    def set_model_config(self):
        self.config = self.get_model_config_df()

    def get_preprocessing_for_the_model_kind(self, kind):
        if kind == "linear":
            ct_num = ColumnTransformer([
                ('scaler', StandardScaler(), self.num_col)
            ])
            ct_cat = ColumnTransformer([
                ("oneHotEncoder", OneHotEncoder(drop="first"), self.cat_col)
            ])
            return "linear_process", FeatureUnion([("ctScaler", ct_num), ("ctOneHotEncoder", ct_cat)])
        elif kind == "tree":
            return "multiEncoder", LabelEncodingColumns(cols=self.cat_col)
        else:
            return -1

    def processing(self, scoring_optimization=None, cv=3):
        config = self.config
        resultat_model = {"cv_results": {}, "best_estimator": {}, "performance_of_best_estimator": {}, "y_predit": {},
                          "scorer": {}, "scoring": {}}
        self.num_col = self.X_train.select_dtypes(exclude=[object]).columns.tolist()
        self.cat_col = self.X_train.select_dtypes(include=[object]).columns.tolist()
        imput = CustomImputer()
        X_train = imput.fit_transform(self.X_train)
        X_test = imput.transform(self.X_test)


        for nom_algo in config.columns.tolist():

            kind = config.loc["_kind", nom_algo]
            tmp_grid = config.loc["_grids", nom_algo]

            pipe_final = Pipeline([self.get_preprocessing_for_the_model_kind(kind), config.loc["_pipeSteps", nom_algo]])
            cv_tmp = GridSearchCV(pipe_final, param_grid=tmp_grid, scoring=scoring_optimization, cv=cv, n_jobs=-1, return_train_score= True)
            cv_tmp.fit(X_train, self.y_train)
            resultat_model["cv_results"][nom_algo] = cv_tmp.cv_results_
            resultat_model["best_estimator"][nom_algo] = cv_tmp.best_estimator_
            resultat_model["performance_of_best_estimator"][nom_algo] = cv_tmp.best_score_
            resultat_model["scorer"][nom_algo] = cv_tmp.scorer_
            resultat_model["scoring"][nom_algo] = cv_tmp.scoring
            y_pred = cv_tmp.predict(X_test)
            resultat_model["y_predit"][nom_algo] = y_pred

        return pd.DataFrame(resultat_model)
