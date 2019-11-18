import pandas as pd
import model
import ml

def main():
    _pb = ml.Classification(name="Loan Prediction",
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
    return res


if __name__ == "__main__":
    to_show = main()
    print(to_show)
    to_show.to_csv(r"C:\Users\Dell\Documents\COURS_TAF\Data\Perso\Loan_prediction\test_generic_ml.csv", index=False)