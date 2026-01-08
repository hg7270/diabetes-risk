Link to deployed webapp: https://diabetes-risk-3mnn8nzoqgfmpyresq8l3x.streamlit.app/

Organization of files:

- __data/*__: all original data files downloaded from public health datasets
- __dataset_and_model_prep/*__: all Jupyter notebooks used to preprocess data and perform model training
  - __preprocess-data.ipynb__: dataset preprocessing, standardization, and merge
  - __prepare-dataset.ipynb__: dataset split into train, test, val sets
  - __train-diabetes-risk.ipynb__: model training and evaluation
- __webapp_files/*__: data files that the webapp depends on
  - __US_FIPS_Codes.xls__: geographical info for counties allowing the map to render
  - __county_data_for_webapp.csv__: county-level data exported after model training
- __webapp.py__: code for webapp
