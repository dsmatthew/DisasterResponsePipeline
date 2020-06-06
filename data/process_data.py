# import libraries
import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load Data function

    Parameters
    ----------
    messages_filepath : str
        File path to the messages.csv file
    categories_filepath : str
        File path to the categories.csv file

    Return
    ------
    df : Pandas DataFrame
        The merged table of both csv files.
    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on=['id'], how='inner')

    return df


def clean_data(df):
    '''
    Clean Data function

    Parameters
    ----------
    df : Pandas DataFrame
        File path to the messages.csv file
    categories_filepath : str
        File path to the categories.csv file

    Return
    ------
    df : Pandas DataFrame
        A cleaned table as merged DataFrame of both input files.
        Cleanup: Splitting up the collection column into individual category columns; Removing duplicates
    '''

    # Exctracting single attributes in collection column
    categories = df['categories'].str.split(';', expand=True)

    # select first row to retrieve the new column names
    row = categories.iloc[0].apply(lambda x: x[0:-2])
    category_colnames = row.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Extract value from each column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Save Data function

    Parameters
    ----------
    df : pandas DataFrame
        preprocessed data which should be stored on disk
    database_filename : str
        File name of the database file

    Return
    ------
    df : Pandas DataFrame
        The merged table of both csv files.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CategorizedMessages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()