import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	'''
	Loads messages.csv and categories.csv and merge them into a pandas.DataFrame.

	Returns a dataframe
    '''

	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	df = messages.merge(categories,on='id')

	return df


def clean_data(df):
	"""
	Cleans data for:
		separating lales into individual columns and fixing the label names
		eliminating duplicated rows
		eliminating rows with conflicting labels

	Returns a dataframe contains cleaned data

	"""

	# create a dataframe of the 36 individual category columns
	categories_fixed = df.categories.str.split(';',expand=True)

   # select the first row of the categories dataframe
	row = categories_fixed.iloc[0]

	# use this row to extract a list of new column names for categories
	category_colnames = row.str.slice(stop=-2).values

	# rename the columns of `categories_fixed`
	categories_fixed.columns = category_colnames

	# concatenate the original dataframe with the new `categories_fixed` dataframe
	df = pd.concat([df,categories_fixed],axis=1)

	# groupby twice to find missmatched rows via the ids 
	df_groupby_idx2 = pd.DataFrame(pd.DataFrame(df.groupby(['id','categories']).message.count()).reset_index().groupby('id').message.count())
	miss_matched_ids = df_groupby_idx2[df_groupby_idx2.message!=1].index.values

	# drop duplicates
	df.drop_duplicates(inplace=True)

	# drop mismatched ids/instances
	df = df[~df.id.isin(miss_matched_ids)]

	return df


def save_data(df, database_filename):
	"""
	Saves data into sqlite database
	"""
	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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