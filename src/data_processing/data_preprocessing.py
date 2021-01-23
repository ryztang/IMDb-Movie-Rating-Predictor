import pandas as pd
import numpy as np

import json


# Expands a column with comma-separated strings into multiple rows (one for each string)
# Other fields are duplicated for the added rows
# This function was obtained from https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-
# entry-to-separate-rows/40449726#40449726
def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
            and len(lst_cols) > 0
            and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    # noinspection PyTypeChecker
    res = (pd.DataFrame({
        col: np.repeat(df[col].values, lens)
        for col in idx_cols},
        index=idx)
           .assign(**{col: np.concatenate(df.loc[lens > 0, col].values)
                      for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
               .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


# Loads data from IMDb datasets and preprocesses the data into a final dataframe to feed the ML model
def load_and_preprocess_data():
    print('Loading titles...')
    titles_df = pd.read_csv('datasets/title_basics.tsv', sep='\t',
                            usecols=['tconst', 'titleType', 'startYear', 'genres'])
    # Only obtain data for movies (not for tv shows, etc.)
    titles_df = titles_df.loc[titles_df['titleType'] == 'movie']

    print('Loading names...')
    names_df = pd.read_csv('datasets/name_basics.tsv', sep='\t', usecols=['nconst', 'primaryName'])
    names_df = names_df.astype({'nconst': 'string', 'primaryName': 'string'})

    print('Loading crew...')
    crew_df = pd.read_csv('datasets/title_crew.tsv', sep='\t')

    print('Loading principals...')
    principals_df = pd.read_csv('datasets/title_principals.tsv', sep='\t',
                                usecols=['tconst', 'nconst', 'category'])
    principals_df = principals_df.loc[principals_df['nconst'] != '\\N']

    print('Loading ratings...')
    ratings_df = pd.read_csv('datasets/title_ratings.tsv', sep='\t', usecols=['tconst', 'averageRating'])

    with open("config.json") as config_file:
        config = json.load(config_file)

    print("Data loaded")
    print("Preprocessing data...")

    # Merge titles dataframe with ratings dataframe and explode genres
    title_ratings_df = titles_df.merge(ratings_df, on='tconst')
    title_ratings_df = explode(title_ratings_df.assign(genres=title_ratings_df.genres.str.split(',')), 'genres')

    # Merge title_ratings_df with principals and crew, and concatenate the two

    title_ratings_principals_df = title_ratings_df.merge(principals_df, on='tconst')

    title_ratings_crew_df = title_ratings_df.merge(crew_df, on='tconst')

    title_ratings_directors_df = title_ratings_crew_df.loc[:, title_ratings_crew_df.columns != 'writers'].copy()
    title_ratings_directors_df = title_ratings_directors_df.loc[title_ratings_directors_df['directors'] != '\\N']
    title_ratings_directors_df = explode(title_ratings_directors_df
                                         .assign(directors=title_ratings_directors_df.directors.str.split(',')),
                                         'directors')
    title_ratings_directors_df = title_ratings_directors_df.rename(columns={'directors': 'nconst'})
    title_ratings_directors_df['category'] = 'director'

    title_ratings_writers_df = title_ratings_crew_df.loc[:, title_ratings_crew_df.columns != 'directors'].copy()
    title_ratings_writers_df = title_ratings_writers_df.loc[title_ratings_writers_df['writers'] != '\\N']
    title_ratings_writers_df = explode(title_ratings_writers_df
                                       .assign(writers=title_ratings_writers_df.writers.str.split(',')),
                                       'writers')
    title_ratings_writers_df = title_ratings_writers_df.rename(columns={'writers': 'nconst'})
    title_ratings_writers_df['category'] = 'writer'

    title_ratings_personnel_df = pd.concat([title_ratings_principals_df, title_ratings_directors_df]) \
        .drop_duplicates().reset_index(drop=True)

    title_ratings_personnel_df = pd.concat([title_ratings_personnel_df, title_ratings_writers_df]) \
        .drop_duplicates().reset_index(drop=True)

    # Modify final_df column types and fill in empty cells

    final_df = title_ratings_personnel_df
    # Round ratings to achieve better model accuracy
    final_df.averageRating = final_df.averageRating.round()
    final_df = final_df.astype({'averageRating': 'int'})

    final_df = final_df.fillna(value='\\N')
    final_df['startYear'] = final_df['startYear'].replace(to_replace='\\N', value=-1)

    final_df = final_df.astype({'tconst': 'string', 'genres': 'string', 'nconst': 'string', 'category': 'string',
                                'startYear': 'int'})

    # Only keep rows that contain director, writer, actor, or actress data
    final_df = final_df.loc[final_df['category'].isin(['director', 'writer', 'actor', 'actress'])]
    final_df = final_df[['tconst', 'nconst', 'category', 'startYear', 'genres', 'averageRating']]

    # Sample size of final_df set in config.json
    sample_size = config['sample_size']
    print('Total number of records: ' + str(len(final_df.index)))
    print('Sample number of records: ' + str(sample_size))
    final_df = final_df.sample(n=sample_size).reset_index(drop=True)

    print("Final dataframe created")
    return final_df, names_df
