import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how="left", on="id")
    return df


def clean_data(df):
    df_cat = df.categories.str.split(pat=";", expand=True)

    row = df_cat.head(1)
    category_colnames = row.values.tolist()[0]
    category_colnames = [s.replace('-1', '').replace('-0', '') for s in category_colnames]
    
    df_cat.columns = category_colnames
    
    for column in df_cat:
        df_cat[column] = df_cat[column].apply(lambda x: x[-1])
        df_cat[column] = df_cat[column].astype(int)

    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, df_cat], axis=1, join="inner")
    df = df.drop_duplicates()
    df.replace(2, 1, inplace=True)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('disaster_messages_table', engine, index=False)
    
    #pass  


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()