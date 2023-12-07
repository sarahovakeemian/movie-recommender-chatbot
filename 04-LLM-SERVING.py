# Databricks notebook source
# MAGIC %md
# MAGIC # 04 LLM SERVING (Llama-2-7B-Chat)
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window up to 4096 tokens. [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept the license terms. Requests will be processed in 1-2 days. Additionally, you will have to request access on Hugging Face.

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
# MAGIC %pip install safetensors
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
import torch
import transformers
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from huggingface_hub import snapshot_download, login
import pandas as pd
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

# MAGIC %run ./resources/variables

# COMMAND ----------

# Login to Huggingface to get access to the model
hf_token = dbutils.secrets.get(f"{secrets_scope}", f"{secrets_hf_key_name}")
login(token=hf_token)

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main

model_id = "meta-llama/Llama-2-7b-chat-hf" # official version, gated (needs login to Hugging Face)
revision = "c1b0db933684edbfe29a06fa47eb19cc48025e93"

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
#ignoring .bin files so that we only grab safetensors
snapshot_location = snapshot_download(repo_id=model_id, revision=revision, cache_dir="/local_disk0/.cache/huggingface/", ignore_patterns=["*.bin"])

# COMMAND ----------

# You can set the experiment name if you want, otherwise it will use the Notebook name
# mlflow.set_experiment('/Users/{}/shovakeemian-llama2-7b-hf-chat'.format(<experiment_name>))

# Define PythonModel to log with mlflow.pyfunc.log_model
class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            use_safetensors=True)
        self.model.eval()

    def _build_prompt(self, system_prompt, instruction):
        """
        This method generates the prompt for the model.
        """
        return f"""<s>[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

    def _generate_response(self, system_prompt, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(system_prompt, prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          system_prompt=model_input['system_prompt'][i]
          temperature = model_input.get("temperature", [1.0])[i]
          max_new_tokens = model_input.get("max_new_tokens", [100])[i]

          outputs.append(self._generate_response(system_prompt, prompt, temperature, max_new_tokens))
      
        return outputs

# COMMAND ----------

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.string, "system_prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_new_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


# Define input example
input_example=pd.DataFrame({
            "prompt":["what is cystic fibrosis (CF)?"], 
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "temperature": [0.1],
            "max_new_tokens": [75]})

# COMMAND ----------

# Log the model with its details such as artifacts, pip requirements and input example
# This may take a couple of minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        registered_model_name=registered_llm_model_name,
        python_model=Llama2(),
        artifact_path="model",
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "safetensors"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

#get the latest version of model
mlflow_client = MlflowClient()

def get_latest_model_version(mlflow_client, model_name):
  model_version_infos = mlflow_client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

model_version=get_latest_model_version(mlflow_client, registered_llm_model_name)
print(model_version)

# COMMAND ----------

# gather other inputs the API needs - they are used as environment variables in the
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

def endpoint_exists(serving_endpoint_name):
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint(serving_endpoint_name):
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint(serving_endpoint_name, served_models):
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {serving_endpoint_name}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": serving_endpoint_name, "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")
  
def update_endpoint(serving_endpoint_name, served_models):
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {serving_endpoint_name}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")

# COMMAND ----------

#GPU Serving of UC pyfunc model 
served_models = [
    {
      "name": llm_model_name,
      "model_name": registered_llm_model_name,
      "model_version": model_version,
      "workload_size": "Small",
      "workload_type": "GPU_MEDIUM",
      "scale_to_zero_enabled": False
    }
]
traffic_config = {"routes": [{"served_model_name": llm_model_name, "traffic_percentage": "100"}]}

# Create or update model serving endpoint
if not endpoint_exists(llm_endpoint_name):
  create_endpoint(llm_endpoint_name, served_models)
else:
  update_endpoint(llm_endpoint_name, served_models)

# COMMAND ----------


