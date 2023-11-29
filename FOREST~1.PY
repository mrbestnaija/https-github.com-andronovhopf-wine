#!/usr/bin/env python
# coding: utf-8

# # Week 3 Assignment:  Data Pipeline Components for Production ML
# 
# In this last graded programming exercise of the course, you will put together all the lessons we've covered so far to handle the first three steps of a production machine learning project - Data ingestion, Data Validation, and Data Transformation.
# 
# Specifically, you will build the production data pipeline by:
# 
# *   Performing feature selection
# *   Ingesting the dataset
# *   Generating the statistics of the dataset
# *   Creating a schema as per the domain knowledge
# *   Creating schema environments
# *   Visualizing the dataset anomalies
# *   Preprocessing, transforming and engineering your features
# *   Tracking the provenance of your data pipeline using ML Metadata
# 
# Most of these will look familiar already so try your best to do the exercises by recall or browsing the documentation. If you get stuck however, you can review the lessons in class and the ungraded labs. 
# 
# Let's begin!

# In[ ]:


# IMPORTANT: This will check your notebook's metadata for grading.
# Please do not continue the lab unless the output of this cell tells you to proceed. 
get_ipython().system('python add_metadata.py --filename C2W3_Assignment.ipynb')


# _**NOTE:** To prevent errors from the autograder, you are not allowed to edit or delete non-graded cells in this notebook . Please only put your solutions in between the `### START CODE HERE` and `### END CODE HERE` code comments, and also refrain from adding any new cells. **Once you have passed this assignment** and want to experiment with any of the non-graded code, you may follow the instructions at the bottom of this notebook._

# ## Table of Contents
# 
# - [1 - Imports](#1)
# - [2 - Load the Dataset](#2)
# - [3 - Feature Selection](#4)
#   - [Exercise 1 - Feature Selection](#ex-1)
# - [4 - Data Pipeline](#4)
#   - [4.1 - Setup the Interactive Context](#4-1)
#   - [4.2 - Generating Examples](#4-2)
#     - [Exercise 2 - ExampleGen](#ex-2)
#   - [4.3 - Computing Statistics](#4-3)
#     - [Exercise 3 - StatisticsGen](#ex-3)
#   - [4.4 - Inferring the Schema](#4-4)
#     - [Exercise 4 - SchemaGen](#ex-4)
#   - [4.5 - Curating the Schema](#4-5)
#     - [Exercise 5 - Curating the Schema](#ex-5)
#   - [4.6 - Schema Environments](#4-6)
#     - [Exercise 6 - Define the serving environment](#ex-6)
#   - [4.7 - Generate new statistics using the updated schema](#4-7)
#       - [Exercise 7 - ImportSchemaGen](#ex-7)
#       - [Exercise 8 - StatisticsGen with the new schema](#ex-8)
#   - [4.8 - Check anomalies](#4-8)
#       - [Exercise 9 - ExampleValidator](#ex-9)
#   - [4.9 - Feature Engineering](#4-9)
#       - [Exercise 10 - preprocessing function](#ex-10)
#       - [Exercise 11 - Transform](#ex-11)
# - [5 - ML Metadata](#5)
#   - [5.1 - Accessing stored artifacts](#5-1)
#   - [5.2 - Tracking artifacts](#5-2)
#     - [Exercise 12 - Get parent artifacts](#ex-12)

# <a name='1'></a>
# ## 1 - Imports

# In[ ]:


# grader-required-cell

import tensorflow as tf
from tfx import v1 as tfx

# TFX libraries
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# For performing feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# For feature visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# Utilities
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf.json_format import MessageToDict
from  tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
import tensorflow_transform.beam as tft_beam
import os
import pprint
import tempfile
import pandas as pd

# To ignore warnings from TF
tf.get_logger().setLevel('ERROR')

# For formatting print statements
pp = pprint.PrettyPrinter()

# Display versions of TF and TFX related packages
print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))
print('TensorFlow Data Validation version: {}'.format(tfdv.__version__))
print('TensorFlow Transform version: {}'.format(tft.__version__))


# <a name='2'></a>
# ## 2 - Load the dataset
# 
# You are going to use a variant of the [Cover Type](https://archive.ics.uci.edu/ml/datasets/covertype) dataset. This can be used to train a model that predicts the forest cover type based on cartographic variables. You can read more about the *original* dataset [here](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info) and we've outlined the data columns below:
# 
# | Column Name | Variable Type | Units / Range | Description |
# | --------- | ------------ | ----- | ------------------- |
# | Elevation | quantitative |meters | Elevation in meters |
# | Aspect | quantitative | azimuth | Aspect in degrees azimuth |
# | Slope | quantitative | degrees | Slope in degrees |
# | Horizontal_Distance_To_Hydrology | quantitative | meters | Horz Dist to nearest surface water features |
# | Vertical_Distance_To_Hydrology | quantitative | meters | Vert Dist to nearest surface water features |
# | Horizontal_Distance_To_Roadways | quantitative | meters | Horz Dist to nearest roadway |
# | Hillshade_9am | quantitative | 0 to 255 index | Hillshade index at 9am, summer solstice |
# | Hillshade_Noon | quantitative | 0 to 255 index | Hillshade index at noon, summer soltice |
# | Hillshade_3pm | quantitative | 0 to 255 index | Hillshade index at 3pm, summer solstice |
# | Horizontal_Distance_To_Fire_Points | quantitative | meters | Horz Dist to nearest wildfire ignition points |
# | Wilderness_Area (4 binary columns) | qualitative | 0 (absence) or 1 (presence) | Wilderness area designation |
# | Soil_Type (40 binary columns) | qualitative | 0 (absence) or 1 (presence) | Soil Type designation |
# | Cover_Type (7 types) | integer | 1 to 7 | Forest Cover Type designation |
# 
# As you may notice, the qualitative data has already been one-hot encoded (e.g. `Soil_Type` has 40 binary columns where a `1` indicates presence of a feature). For learning, we will use a modified version of this dataset that shows a more raw format. This will let you practice your skills in handling different data types. You can see the code for preparing the dataset [here](https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/datasets/covertype/wrangle/prepare.ipynb) if you want but it is **not required for this assignment**. The main changes include:
# 
# * Converting `Wilderness_Area` and `Soil_Type` to strings.
# * Converting the `Cover_Type` range to [0, 6]
# 
# Run the next cells to load the **modified** dataset to your workspace. 

# In[ ]:


# # OPTIONAL: Just in case you want to restart the lab workspace *from scratch*, you
# # can uncomment and run this block to delete previously created files and
# # directories. 

# !rm -rf pipeline
# !rm -rf data


# In[ ]:


# grader-required-cell

# Declare paths to the data
DATA_DIR = './data'
TRAINING_DIR = f'{DATA_DIR}/training'
TRAINING_DATA = f'{TRAINING_DIR}/dataset.csv'

# Create the directory
get_ipython().system('mkdir -p {TRAINING_DIR}')


# In[ ]:


# download the dataset
get_ipython().system('wget -nc https://storage.googleapis.com/mlep-public/course_2/week3/dataset.csv -P {TRAINING_DIR}')


# <a name='3'></a>
# ## 3 - Feature Selection
# 
# For your first task, you will reduce the number of features to feed to the model. As mentioned in Week 2, this will help reduce the complexity of your model and save resources while training. Let's assume that you already have a baseline model that is trained on all features and you want to see if reducing the number of features will generate a better model. You will want to select a subset that has great predictive value to the label (in this case the `Cover_Type`). Let's do that in the following cells.
# 

# In[ ]:


# grader-required-cell

# Load the dataset to a dataframe
df = pd.read_csv(TRAINING_DATA)

# Preview the dataset
df.head()


# In[ ]:


# Show the data type of each column
df.dtypes


# Looking at the data types of each column and the dataset description at the start of this notebook, you can see that most of the features are numeric and only two are not. This needs to be taken into account when selecting the subset of features because numeric and categorical features are scored differently. Let's create a temporary dataframe that only contains the numeric features so we can use it in the next sections.

# In[ ]:


# grader-required-cell

# Copy original dataset
df_num = df.copy()

# Categorical columns
cat_columns = ['Wilderness_Area', 'Soil_Type']

# Label column
label_column = ['Cover_Type']

# Drop the categorical and label columns
df_num.drop(cat_columns, axis=1, inplace=True)
df_num.drop(label_column, axis=1, inplace=True)

# Preview the resuls
df_num.head()


# You will use scikit-learn's built-in modules to perform [univariate feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) on our dataset's numeric attributes. First, you need to prepare the input and target features:

# In[ ]:


# grader-required-cell

# Set the target values
y = df[label_column].values

# Set the input values
X = df_num.values


# Afterwards, you will use [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) to score each input feature against the target variable. Be mindful of the scoring function to pass in and make sure it is appropriate for the input (numeric) and target (categorical) values.

# <a name='ex-1'></a>
# ### Exercise 1: Feature Selection
# 
# Complete the code below to select the top 8 features of the numeric columns.

# In[ ]:


# grader-required-cell

### START CODE HERE ###

# Create SelectKBest object using f_classif (ANOVA statistics) for 8 classes
select_k_best = None

# Fit and transform the input data using select_k_best
X_new = None

# Extract the features which are selected using get_support API
features_mask = None

### END CODE HERE ###

# Print the results
reqd_cols = pd.DataFrame({'Columns': df_num.columns, 'Retain': features_mask})
print(reqd_cols)


# **Expected Output:**
# 
# ```
#                               Columns  Retain
# 0                           Elevation    True
# 1                              Aspect   False
# 2                               Slope    True
# 3    Horizontal_Distance_To_Hydrology    True
# 4      Vertical_Distance_To_Hydrology    True
# 5     Horizontal_Distance_To_Roadways    True
# 6                       Hillshade_9am    True
# 7                      Hillshade_Noon    True
# 8                       Hillshade_3pm   False
# 9  Horizontal_Distance_To_Fire_Points    True
# ```

# If you got the expected results, you can now select this subset of features from the original dataframe and save it to a new directory in your workspace.

# In[ ]:


# grader-required-cell

# Set the paths to the reduced dataset
TRAINING_DIR_FSELECT = f'{TRAINING_DIR}/fselect'
TRAINING_DATA_FSELECT = f'{TRAINING_DIR_FSELECT}/dataset.csv'

# Create the directory
get_ipython().system('mkdir -p {TRAINING_DIR_FSELECT}')


# In[ ]:


# grader-required-cell

# Get the feature names from SelectKBest
feature_names = list(df_num.columns[features_mask])

# Append the categorical and label columns
feature_names = feature_names + cat_columns + label_column

# Select the selected subset of columns
df_select = df[feature_names]

# Write CSV to the created directory
df_select.to_csv(TRAINING_DATA_FSELECT, index=False)

# Preview the results
df_select.head()


# <a name='4'></a>
# ## 4 - Data Pipeline
# 
# With the selected subset of features prepared, you can now start building the data pipeline. This involves ingesting, validating, and transforming your data. You will be using the TFX components you've already encountered in the ungraded labs and you can look them up here in the [official documentation](https://www.tensorflow.org/tfx/api_docs/python/tfx/components).

# <a name='4-1'></a>
# ### 4.1 - Setup the Interactive Context
# 
# As usual, you will first setup the Interactive Context so you can manually execute the pipeline components from the notebook. You will save the sqlite database in a pre-defined directory in your workspace. Please do not modify this path because you will need this in a later exercise involving ML Metadata.

# In[ ]:


# grader-required-cell

# Location of the pipeline metadata store
PIPELINE_DIR = './pipeline'

# Declare the InteractiveContext and use a local sqlite file as the metadata store.
context = InteractiveContext(pipeline_root=PIPELINE_DIR)


# <a name='4-2'></a>
# ### 4.2 - Generating Examples
# 
# The first step in the pipeline is to ingest the data. Using [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen), you can convert raw data to TFRecords for faster computation in the later stages of the pipeline.

# <a name='ex-2'></a>
# #### Exercise 2: ExampleGen
# 
# Use `ExampleGen` to ingest the dataset we loaded earlier. Some things to note:
# 
# * The input is in CSV format so you will need to use the appropriate type of `ExampleGen` to handle it. 
# * This function accepts a *directory* path to the training data and not the CSV file path itself. 
# 
# This will take a couple of minutes to run.

# In[ ]:


# # NOTE: Uncomment and run this if you get an error saying there are different 
# # headers in the dataset. This is usually because of the notebook checkpoints saved in 
# # that folder.
# !rm -rf {TRAINING_DIR}/.ipynb_checkpoints
# !rm -rf {TRAINING_DIR_FSELECT}/.ipynb_checkpoints
# !rm -rf {SERVING_DIR}/.ipynb_checkpoints


# In[ ]:


# grader-required-cell

### START CODE HERE

# Instantiate ExampleGen with the input CSV dataset
example_gen = None

# Run the component using the InteractiveContext instance
None

### END CODE HERE


# <a name='4-3'></a>
# ### 4.3 - Computing Statistics
# 
# Next, you will compute the statistics of your data. This will allow you to observe and analyze characteristics of your data through visualizations provided by the integrated [FACETS](https://pair-code.github.io/facets/) library.

# <a name='ex-3'></a>
# #### Exercise 3: StatisticsGen
# 
# Use [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) to compute the statistics of the output examples of `ExampleGen`. 

# In[ ]:


# grader-required-cell

### START CODE HERE

# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = None
    

# Run the component
None
### END CODE HERE


# In[ ]:


# Display the results
context.show(statistics_gen.outputs['statistics'])


# Once you've loaded the display, you may notice that the `zeros` column for `Cover_type` is highlighted in red. The visualization is letting us know that this might be a potential issue. In our case though, we know that the `Cover_Type` has a range of [0, 6] so having zeros in this column is something we expect.

# <a name='4-4'></a>
# ### 4.4 - Inferring the Schema
# 
# You will need to create a schema to validate incoming datasets during training and serving. Fortunately, TFX allows you to infer a first draft of this schema with the [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) component.

# <a name='ex-4'></a>
# #### Exercise 4: SchemaGen
# 
# Use `SchemaGen` to infer a schema based on the computed statistics of `StatisticsGen`.

# In[ ]:


# grader-required-cell

### START CODE HERE
# Instantiate SchemaGen with the output statistics from the StatisticsGen
schema_gen = None
    
    

# Run the component
None
### END CODE HERE


# In[ ]:


# Visualize the output
context.show(schema_gen.outputs['schema'])


# <a name='4-5'></a>
# ### 4.5 - Curating the schema
# 
# You can see that the inferred schema is able to capture the data types correctly and also able to show the expected values for the qualitative (i.e. string) data. You can still fine-tune this however. For instance, we have features where we expect a certain range:
# 
# * `Hillshade_9am`: 0 to 255
# * `Hillshade_Noon`: 0 to 255
# * `Slope`: 0 to 90
# * `Cover_Type`:  0 to 6
# 
# You want to update your schema to take note of these so the pipeline can detect if invalid values are being fed to the model.

# <a name='ex-5'></a>
# #### Exercise 5: Curating the Schema
# 
# Use [TFDV](https://www.tensorflow.org/tfx/data_validation/get_started) to update the inferred schema to restrict a range of values to the features mentioned above.
# 
# Things to note:
# * You can use [tfdv.set_domain()](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/set_domain) to define acceptable values for a particular feature.
# * These should still be INT types after making your changes.
# * Declare `Cover_Type` as a *categorical* variable. Unlike the other four features, the integers 0 to 6 here correspond to a designated label and not a quantitative measure. You can look at the available flags for `set_domain()` in the official doc to know how to set this.

# In[ ]:


# grader-required-cell

try:
    # Get the schema uri
    schema_uri = schema_gen.outputs['schema']._artifacts[0].uri
    
# for grading since context.run() does not work outside the notebook
except IndexError:
    print("context.run() was no-op")
    schema_path = './pipeline/SchemaGen/schema'
    dir_id = os.listdir(schema_path)[0]
    schema_uri = f'{schema_path}/{dir_id}'


# In[ ]:


# grader-required-cell

# Get the schema pbtxt file from the SchemaGen output
schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))


# In[ ]:


# grader-required-cell

### START CODE HERE ###

# Set the two `Hillshade` features to have a range of 0 to 255
tfdv.set_domain(None, None, schema_pb2.IntDomain(name='Hillshade_9am', min=None, max=None))
tfdv.set_domain(None, None, schema_pb2.IntDomain(name='Hillshade_Noon', min=None, max=None))

# Set the `Slope` feature to have a range of 0 to 90
tfdv.set_domain(None, None, schema_pb2.IntDomain(name='Slope', min=None, max=None))

# Set `Cover_Type` to categorical having minimum value of 0 and maximum value of 6
tfdv.set_domain(None, None, schema_pb2.IntDomain(name='Cover_Type', min=None, max=None, is_categorical=None))

### END CODE HERE ###

tfdv.display_schema(schema=schema)


# You should now see the ranges you declared in the `Domain` column of the schema.

# <a name='4-6'></a>
# ### 4.6 - Schema Environments
# 
# In supervised learning, we train the model to make predictions by feeding a set of features with its corresponding label. Thus, our training dataset will have both the input features and label, and the schema is configured to detect these. 
# 
# However, after training and you serve the model for inference, the incoming data will no longer have the label. This will present problems when validating the data using the current version of the schema. Let's demonstrate that in the following cells. You will simulate a serving dataset by getting subset of the training set and dropping the label column (i.e. `Cover_Type`). Afterwards, you will validate this serving dataset using the schema you curated earlier.

# In[ ]:


# grader-required-cell

# Declare paths to the serving data
SERVING_DIR = f'{DATA_DIR}/serving'
SERVING_DATA = f'{SERVING_DIR}/serving_dataset.csv'

# Create the directory
get_ipython().system('mkdir -p {SERVING_DIR}')


# In[ ]:


# grader-required-cell

# Read a subset of the training dataset
serving_data = pd.read_csv(TRAINING_DATA, nrows=100)

# Drop the `Cover_Type` column
serving_data.drop(columns='Cover_Type', inplace=True)

# Save the modified dataset
serving_data.to_csv(SERVING_DATA, index=False)

# Delete unneeded variable from memory
del serving_data


# In[ ]:


# grader-required-cell

# Declare StatsOptions to use the curated schema
stats_options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)

# Compute the statistics of the serving dataset
serving_stats = tfdv.generate_statistics_from_csv(SERVING_DATA, stats_options=stats_options)

# Detect anomalies in the serving dataset
anomalies = tfdv.validate_statistics(serving_stats, schema=schema)

# Display the anomalies detected
tfdv.display_anomalies(anomalies)


# As expected, the missing column is flagged. To fix this, you need to configure the schema to detect when it's being used for training or for inference / serving. You can do this by setting [schema environments](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic#schema_environments).

# <a name='ex-6'></a>
# #### Exercise 6: Define the serving environment
# 
# Complete the code below to ignore the `Cover_Type` feature when validating in the *SERVING* environment.

# In[ ]:


# grader-required-cell

schema.default_environment.append('TRAINING')

### START CODE HERE ###
# Hint: Create another default schema environment with name SERVING (pass in a string)
schema.default_environment.append(None)

# Remove Cover_Type feature from SERVING using TFDV
# Hint: Pass in the strings with the name of the feature and environment 
tfdv.get_feature(schema, None).not_in_environment.append(None)
### END CODE HERE ###


# If done correctly, running the cell below should show *No Anomalies*.

# In[ ]:


# grader-required-cell

# Validate the serving dataset statistics in the `SERVING` environment
anomalies = tfdv.validate_statistics(serving_stats, schema=schema, environment='SERVING')

# Display the anomalies detected
tfdv.display_anomalies(anomalies)


# We can now save this curated schema in a local directory so we can import it to our TFX pipeline.

# In[ ]:


# grader-required-cell

# Declare the path to the updated schema directory
UPDATED_SCHEMA_DIR = f'{PIPELINE_DIR}/updated_schema'

# Create the said directory
get_ipython().system('mkdir -p {UPDATED_SCHEMA_DIR}')

# Declare the path to the schema file
schema_file = os.path.join(UPDATED_SCHEMA_DIR, 'schema.pbtxt')

# Save the curated schema to the said file
tfdv.write_schema_text(schema, schema_file)


# As a sanity check, let's display the schema we just saved and verify that it contains the changes we introduced. It should still show the ranges in the `Domain` column and there should be two environments available.

# In[ ]:


# grader-required-cell

# Load the schema from the directory we just created
new_schema = tfdv.load_schema_text(schema_file)

# Display the schema. Check that the Domain column still contains the ranges.
tfdv.display_schema(schema=new_schema)


# In[ ]:


# The environment list should show `TRAINING` and `SERVING`.
new_schema.default_environment


# <a name='4-7'></a>
# ### 4.7 - Generate new statistics using the updated schema
# 
# You will now compute the statistics using the schema you just curated. Remember though that TFX components interact with each other by getting artifact information from the metadata store. So you first have to import the curated schema file into ML Metadata. You will do that by using an [ImportSchemaGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportSchemaGen) to create an artifact representing the curated schema.

# <a name='ex-7'></a>
# #### Exercise 7: ImportSchemaGen
# 
# Complete the code below to create a `Schema` artifact that points to the path of the curated schema file.

# In[ ]:


# grader-required-cell

### START CODE HERE ###

# Use ImportSchemaGen to put the curated schema to ML Metadata
user_schema_importer = None
    


# Run the component
context.run(None, enable_cache=False)

### END CODE HERE ###

context.show(user_schema_importer.outputs['schema'])


# With the artifact successfully created, you can now use `StatisticsGen` and pass in a `schema` parameter to use the curated schema.
# 
# <a name='ex-8'></a>
# #### Exercise 8: Statistics with the new schema
# 
# Use `StatisticsGen` to compute the statistics with the schema you updated in the previous section.

# In[ ]:


# grader-required-cell

### START CODE HERE ###
# Use StatisticsGen to compute the statistics using the curated schema
statistics_gen_updated = None
    
    
    


# Run the component
None
### END CODE HERE ###


# In[ ]:


context.show(statistics_gen_updated.outputs['statistics'])


# The chart will look mostly the same from the previous runs but you can see that the `Cover Type` is now under the categorical features. That shows that `StatisticsGen` is indeed using the updated schema.

# <a name='4-8'></a>
# ### 4.8 - Check anomalies
# 
# You will now check if the dataset has any anomalies with respect to the schema. You can do that easily with the [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) component.

# <a name='ex-9'></a>
# #### Exercise 9: ExampleValidator
# 
# Check if there are any anomalies using `ExampleValidator`. You will need to pass in the updated statistics and schema from the previous sections.

# In[ ]:


# grader-required-cell

### START CODE HERE ###

example_validator = None
    
    

# Run the component.
None

### END CODE HERE ###


# In[ ]:


# Visualize the results
context.show(example_validator.outputs['anomalies'])


# <a name='4-10'></a>
# ### 4.10 - Feature engineering
# 
# You will now proceed to transforming your features to a form suitable for training a model. This can include several methods such as scaling and converting strings to vocabulary indices. It is important for these transformations to be consistent across your training data, and also for the serving data when the model is deployed for inference. TFX ensures this by generating a graph that will process incoming data both during training and inference.
# 
# Let's first declare the constants and utility function you will use for the exercise.

# In[ ]:


# grader-required-cell

# Set the constants module filename
_cover_constants_module_file = 'cover_constants.py'


# In[ ]:


get_ipython().run_cell_magic('writefile', '{_cover_constants_module_file}', '\nSCALE_MINMAX_FEATURE_KEYS = [\n        "Horizontal_Distance_To_Hydrology",\n        "Vertical_Distance_To_Hydrology",\n    ]\n\nSCALE_01_FEATURE_KEYS = [\n        "Hillshade_9am",\n        "Hillshade_Noon",\n        "Horizontal_Distance_To_Fire_Points",\n    ]\n\nSCALE_Z_FEATURE_KEYS = [\n        "Elevation",\n        "Slope",\n        "Horizontal_Distance_To_Roadways",\n    ]\n\nVOCAB_FEATURE_KEYS = ["Wilderness_Area"]\n\nHASH_STRING_FEATURE_KEYS = ["Soil_Type"]\n\nLABEL_KEY = "Cover_Type"\n\n# Utility function for renaming the feature\ndef transformed_name(key):\n    return key + \'_xf\'')


# Next you will define the `preprocessing_fn` to apply transformations to the features. 

# <a name='ex-10'></a>
# #### Exercise 10: Preprocessing function
# 
# Complete the module to transform your features. Refer to the code comments to get hints on what operations to perform.
# 
# Here are some links to the docs of the functions you will need to complete this function:
# 
# - [`tft.scale_by_min_max`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_by_min_max)
# - [`tft.scale_to_0_1`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_to_0_1)
# - [`tft.scale_to_z_score`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_to_z_score)
# - [`tft.compute_and_apply_vocabulary`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/compute_and_apply_vocabulary)
# - [`tft.hash_strings`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/hash_strings)

# In[ ]:


# grader-required-cell

# Set the transform module filename
_cover_transform_module_file = 'cover_transform.py'


# In[ ]:


get_ipython().run_cell_magic('writefile', '{_cover_transform_module_file}', '\nimport tensorflow as tf\nimport tensorflow_transform as tft\n\nimport cover_constants\n\n_SCALE_MINMAX_FEATURE_KEYS = cover_constants.SCALE_MINMAX_FEATURE_KEYS\n_SCALE_01_FEATURE_KEYS = cover_constants.SCALE_01_FEATURE_KEYS\n_SCALE_Z_FEATURE_KEYS = cover_constants.SCALE_Z_FEATURE_KEYS\n_VOCAB_FEATURE_KEYS = cover_constants.VOCAB_FEATURE_KEYS\n_HASH_STRING_FEATURE_KEYS = cover_constants.HASH_STRING_FEATURE_KEYS\n_LABEL_KEY = cover_constants.LABEL_KEY\n_transformed_name = cover_constants.transformed_name\n\ndef preprocessing_fn(inputs):\n\n    features_dict = {}\n\n    ### START CODE HERE ###\n    for feature in _SCALE_MINMAX_FEATURE_KEYS:\n        data_col = inputs[feature]         \n        # Transform using scaling of min_max function\n        # Hint: Use tft.scale_by_min_max by passing in the respective column\n        # Use the *default* output range of the function\n        features_dict[_transformed_name(feature)] = None\n\n    for feature in _SCALE_01_FEATURE_KEYS:\n        data_col = inputs[feature]         \n        # Transform using scaling of 0 to 1 function\n        # Hint: tft.scale_to_0_1\n        features_dict[_transformed_name(feature)] = None\n\n    for feature in _SCALE_Z_FEATURE_KEYS:\n        data_col = inputs[feature]         \n        # Transform using scaling to z score\n        # Hint: tft.scale_to_z_score\n        features_dict[_transformed_name(feature)] = None\n\n    for feature in _VOCAB_FEATURE_KEYS:\n        data_col = inputs[feature]         \n        # Transform using vocabulary available in column\n        # Hint: Use tft.compute_and_apply_vocabulary\n        features_dict[_transformed_name(feature)] = None\n\n    for feature in _HASH_STRING_FEATURE_KEYS:\n        data_col = inputs[feature]         \n        # Transform by hashing strings into buckets\n        # Hint: Use tft.hash_strings with the param hash_buckets set to 10\n        features_dict[_transformed_name(feature)] = None\n    \n    ### END CODE HERE ###  \n\n    # No change in the label\n    features_dict[_LABEL_KEY] = inputs[_LABEL_KEY]\n\n    return features_dict')


# In[ ]:


# Test your preprocessing_fn

import cover_transform
from testing_values import feature_description, raw_data

# NOTE: These next two lines are for reloading your cover_transform module in case you need to 
# update your initial solution and re-run this cell. Please do not remove them especially if you
# have revised your solution. Else, your changes will not be detected.
import importlib
importlib.reload(cover_transform)

raw_data_metadata = dataset_metadata.DatasetMetadata(schema_utils.schema_from_feature_spec(feature_description))

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, _ = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(cover_transform.preprocessing_fn))

transformed_data, transformed_metadata = transformed_dataset


# In[ ]:


# Test that the transformed data matches the expected output
transformed_data


# **Expected Output:**
# 
# ```
# [{'Cover_Type': 4,
#   'Elevation_xf': 0.0,
#   'Hillshade_9am_xf': 1.0,
#   'Hillshade_Noon_xf': 1.0,
#   'Horizontal_Distance_To_Fire_Points_xf': 1.0,
#   'Horizontal_Distance_To_Hydrology_xf': 1.0,
#   'Horizontal_Distance_To_Roadways_xf': 0.0,
#   'Slope_xf': 0.0,
#   'Soil_Type_xf': 4,
#   'Vertical_Distance_To_Hydrology_xf': 0.5,
#   'Wilderness_Area_xf': 0}]
# ```

# In[ ]:


# Test that the transformed metadata's schema matches the expected output
MessageToDict(transformed_metadata.schema)


# **Expected Output:**
# 
# ```
# {'feature': [{'name': 'Cover_Type',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Elevation_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Hillshade_9am_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Hillshade_Noon_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Horizontal_Distance_To_Fire_Points_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Horizontal_Distance_To_Hydrology_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Horizontal_Distance_To_Roadways_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Slope_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Soil_Type_xf',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Vertical_Distance_To_Hydrology_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'Wilderness_Area_xf',
#    'type': 'INT',
#    'intDomain': {'isCategorical': True},
#    'presence': {'minFraction': 1.0},
#    'shape': {}}]}
# ```

# <a name='ex-11'></a>
# #### Exercise 11: Transform
# 
# Use the [TFX Transform component](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/Transform) to perform the transformations and generate the transformation graph. You will need to pass in the dataset examples, *curated* schema, and the module that contains the preprocessing function.

# In[ ]:


# grader-required-cell

### START CODE HERE ###
# Instantiate the Transform component
transform = None
    
    
    
### END CODE HERE ###

# Run the component
context.run(transform, enable_cache=False)


# Let's inspect a few examples of the transformed dataset to see if the transformations are done correctly.

# In[ ]:


# grader-required-cell

try:
    transform_uri = transform.outputs['transformed_examples'].get()[0].uri

# for grading since context.run() does not work outside the notebook
except IndexError:
    print("context.run() was no-op")
    examples_path = './pipeline/Transform/transformed_examples'
    dir_id = os.listdir(examples_path)[0]
    transform_uri = f'{examples_path}/{dir_id}'


# In[ ]:


# grader-required-cell

# Get the URI of the output artifact representing the transformed examples
train_uri = os.path.join(transform_uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")


# In[ ]:


# grader-required-cell

# import helper function to get examples from the dataset
from util import get_records

# Get 3 records from the dataset
sample_records_xf = get_records(transformed_dataset, 3)

# Print the output
pp.pprint(sample_records_xf)


# <a name='5'></a>
# ## 5 - ML Metadata
# 
# TFX uses [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) under the hood to keep records of artifacts that each component uses. This makes it easier to track how the pipeline is run so you can troubleshoot if needed or want to reproduce results.
# 
# In this final section of the assignment, you will demonstrate going through this metadata store to retrieve related artifacts. This skill is useful for when you want to recall which inputs are fed to a particular stage of the pipeline. For example, you can know where to locate the schema used to perform feature transformation, or you can determine which set of examples were used to train a model.

# You will start by importing the relevant modules and setting up the connection to the metadata store. We have also provided some helper functions for displaying artifact information and you can review its code in the external `util.py` module in your lab workspace.

# In[ ]:


# grader-required-cell

# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from util import display_types, display_artifacts, display_properties

# Get the connection config to connect to the metadata store
connection_config = context.metadata_connection_config

# Instantiate a MetadataStore instance with the connection config
store = mlmd.MetadataStore(connection_config)

# Declare the base directory where All TFX artifacts are stored
base_dir = connection_config.sqlite.filename_uri.split('metadata.sqlite')[0]


# <a name='5-1'></a>
# #### 5.1 -  Accessing stored artifacts
# 
# With the connection setup, you can now interact with the metadata store. For instance, you can retrieve all artifact types stored with the `get_artifact_types()` function. For reference, the API is documented [here](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore).

# In[ ]:


# grader-required-cell

# Get the artifact types
types = store.get_artifact_types()

# Display the results
display_types(types)


# You can also get a list of artifacts for a particular type to see if there are variations used in the pipeline. For example, you curated a schema in an earlier part of the assignment so this should appear in the records. Running the cell below should show at least two rows: one for the inferred schema, and another for the updated schema. If you ran this notebook before, then you might see more rows because of the different schema artifacts saved under the `./SchemaGen/schema` directory.

# In[ ]:


# grader-required-cell

# Retrieve the transform graph list
schema_list = store.get_artifacts_by_type('Schema')

# Display artifact properties from the results
display_artifacts(store, schema_list, base_dir)


# Moreover, you can also get the properties of a particular artifact. TFX declares some properties automatically for each of its components. You will most likely see `name`, `state` and `producer_component` for each artifact type. Additional properties are added where appropriate. For example, a `split_names` property is added in `ExampleStatistics` artifacts to indicate which splits the statistics are generated for.

# In[ ]:


# grader-required-cell

# Get the latest TransformGraph artifact
statistics_artifact = store.get_artifacts_by_type('ExampleStatistics')[-1]

# Display the properties of the retrieved artifact
display_properties(store, statistics_artifact)


# <a name='5-2'></a>
# #### 5.2 - Tracking artifacts
# 
# For this final exercise, you will build a function to return the parent artifacts of a given one. For example, this should be able to list the artifacts that were used to generate a particular `TransformGraph` instance. 

# <a name='ex-12'></a>
# ##### Exercise 12: Get parent artifacts
# 
# Complete the code below to track the inputs of a particular artifact.
# 
# Tips:
# 
# * You may find [get_events_by_artifact_ids()](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore#get_events_by_artifact_ids) and [get_events_by_execution_ids()](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore#get_executions_by_id) useful here. 
# 
# * Some of the methods of the MetadataStore class (such as the two given above) only accepts iterables so remember to convert to a list (or set) if you only have an int (e.g. pass `[x]` instead of `x`).
# 
# 

# In[ ]:


# grader-required-cell

def get_parent_artifacts(store, artifact):

    ### START CODE HERE ###
    
    # Get the artifact id of the input artifact
    artifact_id = None
    
    # Get events associated with the artifact id
    artifact_id_events = None
    
    # From the `artifact_id_events`, get the execution ids of OUTPUT events.
    # Cast to a set to remove duplicates if any.
    execution_id = set(         
        None
        for event in None
        if None == None
    
    )
    
    # Get the events associated with the execution_id
    execution_id_events = None

    # From execution_id_events, get the artifact ids of INPUT events.
    # Cast to a set to remove duplicates if any.
    parent_artifact_ids = set(         
        event.artifact_id
        for event in execution_id_events
        if event.type == metadata_store_pb2.Event.INPUT
    
    )
    
    # Get the list of artifacts associated with the parent_artifact_ids
    parent_artifact_list = None

    ### END CODE HERE ###
    
    return parent_artifact_list


# In[ ]:


# grader-required-cell

# Get an artifact instance from the metadata store
artifact_instance = store.get_artifacts_by_type('TransformGraph')[0]

# Retrieve the parent artifacts of the instance
parent_artifacts = get_parent_artifacts(store, artifact_instance)

# Display the results
display_artifacts(store, parent_artifacts, base_dir)


# **Expected Output:**
# 
# *Note: The ID numbers may differ.*
# 
# | artifact id | type | uri |
# | ----------- | ---- | --- |
# | 1	| Examples | ./CsvExampleGen/examples/1 |
# | 4	| Schema | ./ImportSchemaGen/schema/4 |

# **Congratulations!** You have now completed the assignment for this week. You've demonstrated your skills in selecting features, performing a data pipeline, and retrieving information from the metadata store. Having the ability to put these all together will be critical when working with production grade machine learning projects. For next week, you will work on more data types and see how these can be prepared in an ML pipeline. **Keep it up!**

# <details>
#   <summary><font size="2" color="darkgreen"><b>Please click here if you want to experiment with any of the non-graded code.</b></font></summary>
#     <p><i><b>Important Note: Please only do this when you've already passed the assignment to avoid problems with the autograder.</b></i>
#     <ol>
#         <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “Edit Metadata”</li>
#         <li> Hit the “Edit Metadata” button next to the code cell which you want to lock/unlock</li>
#         <li> Set the attribute value for “editable” to:
#             <ul>
#                 <li> “true” if you want to unlock it </li>
#                 <li> “false” if you want to lock it </li>
#             </ul>
#         </li>
#         <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “None” </li>
#     </ol>
#     <p> Here's a short demo of how to do the steps above: 
#         <br>
#         <img src="https://drive.google.com/uc?export=view&id=14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1" align="center">
# </details>
