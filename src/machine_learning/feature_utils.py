from tensorflow import feature_column


# Define feature columns of dataframe to use in ML model
def define_feature_columns(dataframe):

    print("Defining feature columns...")
    feature_columns = []

    # Create embedding column for name IDs
    name_id = feature_column.categorical_column_with_vocabulary_list(
        'nconst', dataframe.nconst.unique())
    # Dimension set to 30 (approximately fourth root of the number of unique name IDs)
    name_id_embedding = feature_column.embedding_column(name_id, dimension=30)
    feature_columns.append(name_id_embedding)

    # Create indicator columns for category and genres
    indicator_column_names = ['category', 'genres']
    for col_name in indicator_column_names:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name, dataframe[col_name].unique())
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)

    # Create bucketized column for startYear (a.k.a. release date)
    start_year_numeric = feature_column.numeric_column('startYear')
    start_year_bucket = feature_column.bucketized_column(
        start_year_numeric, boundaries=[1927, 1940, 1950, 1960, 1970, 1980, 1990, 1995, 2000, 2005, 2010, 2015])
    feature_columns.append(start_year_bucket)

    print("Feature columns defined")
    return feature_columns
