

# PyCaret Regression App
=========================

A web application built using Dash and PyCaret for predicting smartphone prices based on various features.

## Installation
---------------

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries and dependencies specified in the `requirements.txt` file.

## Running the Application
-------------------------

To run the application, navigate to the root directory of the repository and execute the following command:

```bash
python app.py
```

This will start the Dash development server, and the application will be available at `http://127.0.0.1:8050/` in your web browser.

## Data Preparation
-----------------

To prepare the data for training a new model, you can use the `get_data.py` script to scrape data from a website or use your own dataset. The script uses Selenium and BeautifulSoup to extract data from a webpage.

To train a new model, you can use the `data_cleaning_and_model.ipynb` notebook, which includes data cleaning, feature engineering, and model training using PyCaret.

The application uses a pre-trained regression model stored in the `Models` directory. The model was trained on a dataset of smartphone features and prices.


## Configuration
-------------

The application uses a YAML file `numeric_stats.yaml` to store statistical information about the numeric features.
