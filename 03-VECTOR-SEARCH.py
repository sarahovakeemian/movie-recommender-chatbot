# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC ## Databricks Vector Search
# MAGIC In this notebook we will create a Vector Search Index on top of our Delta Lake table
# MAGIC
# MAGIC We now have our knowledge base ready, and saved as a Delta Lake table within Unity Catalog (including permission, lineage, audit logs and all UC features).
# MAGIC
# MAGIC Typically, deploying a production-grade Vector Search index on top of your knowledge base is a difficult task. You need to maintain a process to capture table changes, index the model, provide a security layer, and all sorts of advanced search capabilities.
# MAGIC
# MAGIC Databricks Vector Search removes those painpoints.
# MAGIC
# MAGIC Databricks Vector Search is a new production-grade service that allows you to store a vector representation of your data, including metadata. It will automatically sync with the source Delta table and keep your index up-to-date without you needing to worry about underlying pipelines or clusters. 
# MAGIC
# MAGIC It makes embeddings highly accessible. You can query the index with a simple API to return the most similar vectors, and can optionally include filters or keyword-based queries.

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#import Vector Search and initiate the class
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# MAGIC %run ./resources/variables

# COMMAND ----------

# We need to make sure that the delta table we are going to use for the delta sync has Change Data Feed enabled to allow the serverless DLT job to update the VS Index with new documents as the delta table is updated. 
spark.sql(f'''
          ALTER TABLE {catalog}.{schema}.{sync_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
          ''')

# COMMAND ----------

#create a Vector Search Index
vsc.create_endpoint(
    name=vs_endpoint_name,
    endpoint_type="STANDARD" #PERFORMANCE_OPTIMIZED, STORAGE_OPTIMIZED
)

# COMMAND ----------

# Use the following code if you need to delete the vs endpoint created.
# vsc.delete_endpoint(f'{vs_endpoint_name}')

# COMMAND ----------

#List Vector Search Endpoints in workspace ( you shoudl see the one you created in previous cell)
vsc.list_endpoints()

# COMMAND ----------

#create a vector search sync with a delta table. This will create a serverless DLT job that will manage creating the embeddings of any new documents that are added to the delta table.
vsc.create_delta_sync_index(
  endpoint_name=vs_endpoint_name,
  source_table_name=sync_table_fullname,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="wikipedia_movie_id",
  embedding_source_column="document",
  embedding_model_endpoint_name=embedding_endpoint_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Let's query our Vector Search Index
# MAGIC
# MAGIC We can add filters to the query. filters for Vector Search are possible and are passed as a dictionary of boolean conditions that have to match for a vector to be returned. The following are supported boolean operators with examples:
# MAGIC
# MAGIC 1. Equality: "id": 10 - id is equal to 10
# MAGIC
# MAGIC 2. Less Than/Greater Than (Equal To):
# MAGIC  - "field_name <": 10 
# MAGIC   - “field_name >": 10
# MAGIC   - "field_name <=": 10
# MAGIC   - "field_name >=": 10
# MAGIC
# MAGIC 3. IN: "field_name": (10, 20) - field_name is either 10 or 20
# MAGIC 4. OR: “id OR author”: [10,”John”] - id is 10 OR author is John.
# MAGIC 5. NOT: “id NOT”: 10 - id is not 10

# COMMAND ----------

#Once our Vector Search Index is created (may take some time depending on how many documents are synced), lets do a similarity search
results = vsc.get_index(index_name=vs_index_fullname, endpoint_name=vs_endpoint_name).similarity_search(
  query_text="romantic comedy set in new york",
  columns=["wikipedia_movie_id", "document", "premium", 'movie_runtime', 'childproof', 'rating'],
  filters= {"premium": 1,
            'movie_runtime >=': 90},
  num_results=10)

# COMMAND ----------

results_columns=["wikipedia_movie_id", "document", "premium", 'movie_runtime', 'childproof', 'rating', 'score']
results_df=spark.createDataFrame(results['result']['data_array'], schema=results_columns)

# COMMAND ----------

display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Adding Documents to our Vector Search Index
# MAGIC
# MAGIC

# COMMAND ----------

# let's take 1000 documents that don't exist in our Vector Search Index and add them to our delta table that is synced with the Vector Search Index
df=spark.sql(f'''
             select 
                wikipedia_movie_id,
                document,
                movie_name, 
                movie_release_date, 
                movie_runtime, 
                childproof, 
                premium, 
                rating, 
                document_num_tokens_llama, 
                document_num_chars,
                document_num_words 
              from {catalog}.{schema}.movie_documents_silver
                except 
              select 
                wikipedia_movie_id, 
                document, 
                movie_name, 
                movie_release_date, 
                movie_runtime, 
                childproof, 
                premium, 
                rating, 
                document_num_tokens_llama, 
                document_num_chars,
                document_num_words 
              from {sync_table_fullname}
            limit 1000;''')

# COMMAND ----------

display(df)

# COMMAND ----------

df.write.mode("append").saveAsTable(f"{sync_table_fullname}")

# COMMAND ----------

#let's sync our Vector Search Index. We can do this programatically through the python SDk or through the UI
vsc.get_index(
    index_name=vs_index_fullname,
    endpoint_name=vs_endpoint_name).sync()
