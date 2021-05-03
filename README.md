# cse6250group
Group project for CS6250 OMSCS


Data Sources: 

ISRUC-SLEEP Dataset - https://sleeptight.isr.uc.pt/

Sleep Heart Health Study - https://sleepdata.org/datasets/shhs

PhysioNet The Sleep-EDF database - https://www.physionet.org/content/sleep-edfx/1.0.0/

Folder Structure:
- Data: Includes analytics and post-processed data for machine learning algorithms
    - Raw data is included as well as the train and test sets for running locally.
- Preprocess: Includes data downloads from various sources and aggregation of data
- Visualization: Includes Tableau notebook used to visualize results as well as relevant output data
- dockerfile: container set up information. Dockerfile based on Pysark notebook from docker stacks. See https://github.com/jupyter/docker-stacks/tree/master/pyspark-notebook

## Running
Set up docker container using environment.yml

## Data Preprocessing
Run notebooks in preprocess folder

## Machine Learning
Run mean_shift.ipynb, kmeans_spark.py, and agglomerative_spark.py using train and test sets in data folder

## Visualization
Run Tableau File in visualization folder.


## Other notes
Please ensure requirements in environment for either local or docker contain match the specifications.
