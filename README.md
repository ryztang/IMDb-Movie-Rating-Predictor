# IMDb-Movie-Rating-Predictor

A Python application that trains a deep neural network to predict IMDb movie ratings based on a movie's directors, writers, actors, actresses, genres, and release date.
There are three main components to this project: data preprocessing, model creation and training, and model prediction.  
  
Data preprocessing involves loading structured data from IMDb datasets and processing data into a final dataframe, which is used as input to train the ML model.  
  
Model creation and training begins with defining the feature columns used in the input layer. These include embedding feature columns (that perform dimensionality reduction), along with bucketized and one-hot encoded categorical feature columns. 2 hidden layers are also created, and the output layer performs multi-class classification to predict a movie's rating. Input is split into training, validation, and testing datasets. These are used to train the model through supervised learning.  
  
Model prediction uses the trained model to predict a final rating for the movie.

Below is a diagram of the data flow.

### Data Flow Diagram:
![Data Flow Diagram](diagrams/Data_Flow_Diagram.png?raw=true)

As shown above, data from IMDb datasets are first preprocessed, then used to train the model.  
Below is a diagram of the relationships between these datasets, including the fields used for this project.  

### IMDb Datasets Diagram:
![New Release DB Diagram](diagrams/IMDb_Datasets_Diagram.png?raw=true)

## Libraries

* pandas and NumPy to preprocess data from IMDb datasets
* TensorFlow and Keras to create and train model
* scikit-learn to split training, validation, and testing datasets

## Data Preprocessing

Movie data is stored in 5 IMDb datasets: title_basics, name_basics, title_crew, title_principals, and title_ratings. These should be saved under a `datasets/` folder under the project's root directory (not included in this repo, see [Additional Notes](#additional-notes)). 
  
Using pandas and numpy, title_basics and title_ratings are first joined to obtain all movies that have a rating. Then, title_crew and title_principals are processed to obtain information about the people who worked on those movies. The preprocessed data is stored in a final dataframe that includes columns for nconst (i.e. name ID), category (e.g. director, writer, etc.), startYear (i.e. release date), genres, and averageRating. Thus, the averageRating in each row is the label for the respective nconst, category, startYear, and genre combination. There are around 3000000 records in total. The sample size of the final dataframe can be adjusted in [config.json](config.json).

name_basics is loaded and used to look up name IDs when predicting ratings.

## Model

### Defining feature columns

Using tensorflow, the feature columns are defined to be nconst, category, startYear, and genre. nconst is defined to be an embedding column due to the large number of unique name IDs (one per person). Category and genre are both defined to be indicator columns (one hot encoding) since they have less possible values. startYear is defined as a bucketized column, with each bucket representing the decade or half a decade the movie came out in.

### Creating and training model

The feature columns are used as the input layer to the neural network. Keras is used to define the layers. There are two hidden layers, both using ReLU activation function. The output layer uses softmax activation function to perform multi-class classification. The averageRating obtained from data preprocessing is rounded to the nearest integer to achieve higher accuracy. Thus, the predicted ratings can be integers from 0 to 10. A total of 5 epochs is currently used. 

Training the model with around 3000000 records (the total number of records in the final dataframe) achieves a training accuracy of ~80% and a validation and testing accuracy of ~65%. Compared to the 1/11, or ~9%, chance of guessing the correct rating (0 to 10) without any training, this result is significantly better.

### Predicting using model

After training the model, the user is prompted for input to predict a random movie's rating. The user is asked for all features (directors, writers, actors, actresses, genres, release date). The user input data is processed and stored in a dataframe with the same columns as the dataframe used to train the model. This is fed as input to the model, which then predicts a rating for every nconst, category, genre, startYear combination. The average of all of these predicted ratings are taken to be the final predicted rating.

## Possible Enhancements

1. The dataframe passed into the model currently splits data for one movie into multiple rows. For example, if there are multiple actors, each actor is given his or her own row and label. When predicting a movie's rating, the average of all predicted labels is taken as the predicted rating. A more precise implementation would allow for multi-value feature columns. For example, a column could be named actors and each movie would have its own row with a list of actors as a feature. However, this was too complicated to implement for this project.
  
2. Tweaking the model, such as adjusting number of nodes per layer, number of layers, or number of epochs, to achieve higher accuracy.
  
## Additional Notes

IMDb datasets were downloaded from https://www.imdb.com/interfaces/. They were not posted in this repo.

Information courtesy of  
IMDb  
(http://www.imdb.com).  
Used with permission.
