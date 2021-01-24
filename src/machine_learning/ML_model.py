import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from src.data_processing.data_preprocessing import explode


# Keeps track of new people who are not in names_df when predicting
new_persons = {}
new_num = 0


# Splits dataframe into training, validation, and test datasets
def create_datasets(dataframe):
    print("Creating datasets...")
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    train_ds = _df_to_dataset(train)
    val_ds = _df_to_dataset(val)
    test_ds = _df_to_dataset(test)
    print("Datasets created")
    return train_ds, val_ds, test_ds


# Defines ML model and trains it using train_ds and val_ds
def create_and_train_model(feature_columns, train_ds, val_ds):

    print("Creating model...")

    # Input layer processing feature column data
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # Define all layers - 2 hidden layers, output layer performs multi-class classification
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.3),
        layers.Dense(11, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5)

    print("Model created and trained")
    return model


# Tests ML model using test_ds
def test_model(model, test_ds):
    print("Testing model...")
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)


# Predicts rating for new movie
def predict(model, names_df):
    print('')
    print('Welcome to IMDb Movie Rating Predictor!')
    print('We need information about your movie to predict the rating\n')

    quit_predict = False

    # Allows multiple predictions with same model
    while not quit_predict:

        print('Enter full names of directors (e.g. Christopher Nolan), each on their own line, then enter "done":')
        directors = _get_personnel(names_df)
        print('')
        print('Enter full names of writers (e.g. Aaron Sorkin), each on their own line, then enter "done":')
        writers = _get_personnel(names_df)
        print('')
        print('Enter full names of actors (e.g. Leonardo DiCaprio), each on their own line, then enter "done":')
        actors = _get_personnel(names_df)
        print('')
        print('Enter full names of actresses (e.g. Margot Robbie), each on their own line, then enter "done":')
        actresses = _get_personnel(names_df)
        print('')

        personnel = directors + writers + actors + actresses

        print('Enter genres, each on its own line, then enter "done":')
        genres = _get_genres()
        print('')

        print('Enter movie release date (if unknown then enter -1):')
        release_date = input()
        release_date = int(release_date)
        print('')

        print('Predicting rating...')

        if not personnel:
            personnel = ['\\N']

        if not genres:
            genres = ['\\N']

        category_column = ['director'] * len(directors) + ['writer'] * len(writers) + ['actor'] * len(actors) + \
                          ['actress'] * len(actresses)
        if not category_column:
            category_column = ['\\N']

        start_year_column = [release_date] * len(personnel)
        genre_temp = ','.join(genres)
        genre_column = [genre_temp] * len(personnel)

        # Create prediction dataframe to feed ML model
        predict_df = pd.DataFrame({'nconst': personnel, 'category': category_column, 'startYear': start_year_column,
                                   'genres': genre_column})

        predict_df = explode(predict_df.assign(genres=predict_df.genres.str.split(',')), 'genres')

        ds = tf.data.Dataset.from_tensor_slices(predict_df.to_dict(orient='list'))
        ds = ds.batch(64)
        predicted_probs = model.predict(ds)

        # Obtain predicted classes for all rows in predict_df
        predicted_ratings = np.argmax(predicted_probs, axis=-1)

        print('')

        # Obtains average of all predicted classes
        predicted_rating = 0
        num_ratings = 0
        for rating in predicted_ratings:
            predicted_rating += rating
            num_ratings += 1
        predicted_rating = float(predicted_rating)
        predicted_rating = predicted_rating / num_ratings
        predicted_rating = round(predicted_rating, 1)
        print('Predicted rating: ' + str(predicted_rating) + '\n')

        print('Predict again (y/n)?')
        predict_again = input()
        if predict_again == 'n':
            quit_predict = True
        print('')

    print('Thanks for using IMDb Movie Rating Predictor!')


# Converts dataframe fed to model into tensorflow dataset
def _df_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop('averageRating')
    ds = tf.data.Dataset.from_tensor_slices((dataframe.to_dict(orient='list'), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(64)
    return ds


# Reads user input for personnel when predicting rating (e.g. director, writer, etc.)
def _get_personnel(names_df):
    global new_persons
    global new_num

    personnel = []
    person = input()
    while person != 'done':
        person_df = names_df.loc[names_df['primaryName'] == person]
        # Check if person is new
        if person_df.empty:
            if person in new_persons:
                personnel.append(new_persons[person])
            else:
                # Create new ID for new person
                new_id = 'nm_new' + str(new_num)
                new_persons[person] = new_id
                personnel.append(new_id)
                new_num += 1
        else:
            # Obtain existing ID for person
            person_row = person_df.iloc[0]
            person_id = person_row['nconst']
            personnel.append(person_id)
        person = input()
    return personnel


# Reads user input for genres when predicting rating
def _get_genres():
    genres = []
    genre = input()
    while genre != 'done':
        genres.append(genre)
        genre = input()
    return genres
