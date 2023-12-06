# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #EMBEDDING MODEL
# MAGIC
# MAGIC The embedding model is what will be used to convert our plot summaries into vector embeddings. These vector embeddings will be loaded into a Vector Search Index and allow for fast "Similar Search". This is a very important part of the RAGs architechture. In this notebook we will be using the [e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) open source embedding model from hugging face. 

# COMMAND ----------

import json
import pandas as pd
import requests
import time
from sentence_transformers import SentenceTransformer

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %run ./resources/variables

# COMMAND ----------

#downloading the embedding model from hugging face
source_model_name = 'intfloat/e5-small-v2'
model = SentenceTransformer(source_model_name)

# COMMAND ----------

# Test the model, just to show it works.
sentences = ["Checking if this works", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)

# COMMAND ----------

# Compute input/output schema.
signature = mlflow.models.signature.infer_signature(sentences, embeddings)
print(signature)

# COMMAND ----------

#register model into UC
model_info = mlflow.sentence_transformers.log_model(
  model,
  artifact_path="model",
  signature=signature,
  input_example=sentences,
  registered_model_name=registered_embedding_model_name)

#write a model description
mlflow_client = mlflow.MlflowClient()

mlflow_client.update_registered_model(
  name=f"{registered_embedding_model_name}",
  description="https://huggingface.co/intfloat/e5-small-v2"
)

# COMMAND ----------

#get latest version of model
def get_latest_model_version(mlflow_client, model_name):
  model_version_infos = mlflow_client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

model_version=get_latest_model_version(mlflow_client, registered_embedding_model_name)

print(model_version)

# COMMAND ----------

#serve embedding model with model serving
deploy_headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}
deploy_url = f'{workspace_url}/api/2.0/serving-endpoints'
endpoint_config = {
  "name": embedding_endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{embedding_model_name}',
      "model_name": registered_embedding_model_name,
      "model_version": model_version,
      "workload_type": "CPU",
      "workload_size": "Medium", #maybe change to Medium
      "scale_to_zero_enabled": False,
    }]
  }
}

endpoint_json = json.dumps(endpoint_config, indent='  ')
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

print(deploy_response.json())

# COMMAND ----------

# Prepare data for query.
#Query endpoint (once ready)
sentences = ['Hello world', 'Good morning']
ds_dict = {'dataframe_split': pd.DataFrame(pd.Series(sentences)).to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)
print(data_json)

# COMMAND ----------

#testing endpoint
invoke_headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}
invoke_url = f'{workspace_url}/serving-endpoints/{embedding_endpoint_name}/invocations'
print(invoke_url)

start = time.time()
invoke_response = requests.request(method='POST', headers=invoke_headers, url=invoke_url, data=data_json, timeout=360)
end = time.time()
print(f'time in seconds: {end-start}')

if invoke_response.status_code != 200:
  raise Exception(f'Request failed with status {invoke_response.status_code}, {invoke_response.text}')

print(invoke_response.text)

# COMMAND ----------


