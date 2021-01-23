from src.data_processing.data_preprocessing import load_and_preprocess_data
from src.machine_learning.feature_utils import define_feature_columns
from src.machine_learning.ML_model import create_datasets, create_and_train_model, test_model, predict


def main():
    dataframe, names_df = load_and_preprocess_data()
    feature_columns = define_feature_columns(dataframe)
    train_ds, val_ds, test_ds = create_datasets(dataframe)
    model = create_and_train_model(feature_columns, train_ds, val_ds)
    test_model(model, test_ds)
    predict(model, names_df)


if __name__ == '__main__':
    main()
