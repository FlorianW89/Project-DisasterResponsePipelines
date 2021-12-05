# Disaster Response Pipeline Project

### Summary:
- In this project, I analyzed disaster data from Figure Eight and built a model for an API that classifies disaster messages.
- The data set contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.
- My project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Explanation:
- ETL Pipeline: In the Python script "process_data.py" the data cleaning pipeline
	Loads the messages and categories datasets
	Merges the two datasets
	Cleans the data
	Stores it in a SQLite database

- ML Pipeline: In the Python script "train_classifier.py" the machine learning pipeline
	Loads data from the SQLite database
	Splits the dataset into training and test sets
	Builds a text processing and machine learning pipeline
	Trains and tunes a model using GridSearchCV
	Outputs results on the test set
	Exports the final model as a pickle file


### Instructions:
1. All files should be saved in the same folder. For the two html-templates create a sub-folder called "templates".

    - To run ETL pipeline that cleans data and stores in database
		python exec_process_data.py disaster_messages.csv disaster_categories.csv disaster_response_database.db
        
    - To run ML pipeline that trains classifier and saves
        python exec_train_classifier.py disaster_response_database.db classifier.pkl

2. Run the following command in the same directory to run your web app.
    python exec_run.py
    
    - Now, open another Terminal Window and type
    	env|grep WORK

	- In a new web browser window, type in the following
    	https://SPACEID-3001.SPACEDOMAIN
		(with the outout from the last command)
	
