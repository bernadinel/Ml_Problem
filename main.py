import pandas as pd
import os
import model
import ml


def main(output_dir="", name="Loan_Prediction", type_ml="clf",
         path_train=r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction\train_ctrUa4K.csv",
         path_test=r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction\test_lAUu6dG.csv",
         id_col="Loan_ID", cible_col="Loan_Status"):
    """

    :param output_dir: Where the result should be stored
    :param name: The name of the problem to solve
    :param type_ml: The type of Machine Learning problem: Classification 'clf' or Regression 'r'
    :param path_train: The path to the csv trainfile
    :param path_test: The path to the csv testfile
    :param id_col: The column of the id
    :param cible_col: The column of the target
    :return:
    """

    _pb = ml.MlProblem(name=name, type_ml=type_ml,
                       path_train=path_train,
                       path_test=path_test,
                       id_col=id_col, cible_col=cible_col)
    global _model
    _model = model.Model(type_ml=_pb.type_ml, X_train=_pb.get_X_train_raw(), y_train=_pb.get_y_train_raw(),
                         X_test=_pb.get_X_to_predict_raw())
    _model.encode_cible_if_classifier()
    _model.set_model_and_params_for_ml()
    _model.set_model_config()
    res = _model.processing()
    res.to_csv(os.path.join(output_dir, "ml_result_" + _pb.name + ".csv"))


if __name__ == "__main__":
    main(output_dir=r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction")
