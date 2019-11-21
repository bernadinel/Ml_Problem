import pandas as pd
import os
import model
import ml


def main(output_dir=""):
    _pb = ml.MlProblem(name="Loan_Prediction", type_ml="clf",
                            path_train=r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction\train_ctrUa4K.csv"
                            , path_test=r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction\test_lAUu6dG.csv"
                            , id_col="Loan_ID", cible_col="Loan_Status")
    global _model
    _model = model.Model(type_ml=_pb.type_ml, X_train=_pb.get_X_train_raw(), y_train=_pb.get_y_train_raw(),
                         X_test=_pb.get_X_to_predict_raw())
    _model.encode_cible_if_classifier()
    _model.set_model_and_params_for_ml()
    _model.set_model_config()
    res = _model.processing()
    res.to_csv(os.path.join(output_dir, "ml_result_" + _pb.name + ".csv"))


if __name__ == "__main__":
    main(r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction")
