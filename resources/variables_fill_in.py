# Databricks notebook source
# MAGIC %md
# MAGIC # Rename this file variables.py after you fill in your variables.

# COMMAND ----------

#GENERAL
catalog=''
schema=''

secrets_scope=''
secrets_hf_key_name=''

workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl") 
base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

#DATA PREP
volume_path="/Volumes/"

chunk_size=200
chunk_overlap=50

sync_table_name = ''

# COMMAND ----------

#EMBEDDING MODEL
embedding_model_name=''
registered_embedding_model_name = f'{catalog}.{schema}.{embedding_model_name}'
embedding_endpoint_name = '' 

# COMMAND ----------

#VECTOR SEARCH
vs_endpoint_name=''

vs_index = ''
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"

sync_table_fullname = f"{catalog}.{schema}.{sync_table_name}"

# COMMAND ----------

#LLM SERVING
llm_model_name=''
registered_llm_model_name=f'{catalog}.{schema}.{llm_model_name}'
llm_endpoint_name = ''
