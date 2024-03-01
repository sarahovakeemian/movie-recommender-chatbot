# Databricks notebook source
# MAGIC %md
# MAGIC #PIP Installs

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC
# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC
# MAGIC %pip install pip mlflow[databricks]==2.9.0
# MAGIC
# MAGIC %pip install google-search-results numexpr
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
import os

from operator import itemgetter

from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda

# from langchain_core.messages import HumanMessage
from langchain.memory import ChatMessageHistory, ConversationSummaryMemory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain, LLMChain, SequentialChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import load_tools
from langchain.agents import initialize_agent


# COMMAND ----------

# MAGIC %run ./resources/variables

# COMMAND ----------

# MAGIC %md
# MAGIC #LCEL

# COMMAND ----------

prompt = PromptTemplate(input_variables=["topic"], template="Tell me a short joke about {topic}")

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({'topic':'cats'})

# COMMAND ----------

# MAGIC %md
# MAGIC #MULTI-CHAIN CHAINS

# COMMAND ----------

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

llm= ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

chain1 = prompt1 | llm | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | llm
    | StrOutputParser()
)

chain2.invoke({"person": "obama", "language": "spanish"})

# COMMAND ----------

# MAGIC %md
# MAGIC #AGENTS

# COMMAND ----------

os.environ["SERPAPI_API_KEY"] ='4cbdc6699f3110c59a0e1189869009cbd2b2846758ce00a591cb24135df724e5'

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200, temperature=1)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)

agent.run("Who is the United States President? How old will the president be in 5 years?")
# agent.run("Who is the current leader of Japan? What is their age divided by 2")

# COMMAND ----------

tools

# COMMAND ----------

# MAGIC %md
# MAGIC #RUNNABLES

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnableParallel

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | llm

poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | llm
)

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

map_chain.invoke({"topic": "bear"})

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnablePassthrough

# COMMAND ----------

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnableLambda

# COMMAND ----------

def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("what is {a} + {b}")
llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)


chain = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | llm
)

chain.invoke({"foo": "test", "bar": "testing"})

# COMMAND ----------

# MAGIC %md
# MAGIC #### RunnableBranch

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)


chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `Langchain`, `Cooking`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)

response=chain.invoke({"question": "how do i unclog a drain"})

# COMMAND ----------

response.lower().strip()

# COMMAND ----------



langchain_chain = (
    PromptTemplate.from_template(
        """You are an expert in langchain. \
Always answer questions starting with "I'm the best at Langchain...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)


chef_chain = (
    PromptTemplate.from_template(
        """You are an expert in cooking. \
Always answer questions starting with "Mama mia, let's get cookin....". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:

Question: {question}
Answer:"""
    )
    | llm
)



branch = RunnableBranch(
    (lambda x: "langchain" in x["topic"].lower().strip(), langchain_chain),
    (lambda x: "cooking" in x["topic"].lower().strip(), chef_chain),
    general_chain,
)


full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch


full_chain.invoke({"question": "how do I make meatloaf?"})   

## Can also do same logic with Runnable Lambda
# def route(info):
#     if "langchain" in info["topic"].lower():
#         return langchain_chain
#     elif "cook" in info["topic"].lower():
#         return chef_chain
#     else:
#         return general_chain
      


# COMMAND ----------

# MAGIC %md
# MAGIC #MEMORY

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.invoke('hi there')


# COMMAND ----------

conversation.invoke('what color is the sky and why?')

# COMMAND ----------

# MAGIC %md
# MAGIC # LET'S BUILD OUR MOVIE RECOMMENDER CHAT BOT IN LANGCHAIN

# COMMAND ----------

vsc = VectorSearchClient()

# COMMAND ----------

# function that returns results from vector search based on query
def find_relevant_doc(question, filters_json={"premium": (0,1)},  num_results = 3, relevant_threshold = 0.70, index_name=vs_index_fullname, endpoint_name=vs_endpoint_name, rating_rerank=True):
    results = vsc.get_index(index_name=vs_index_fullname, endpoint_name=vs_endpoint_name).similarity_search(
      query_text=question,
      columns=["wikipedia_movie_id", "document", "premium", 'movie_runtime', 'childproof', 'movie_name', 'rating'],
      filters=filters_json,
      num_results=num_results)
    
    docs = results.get('result', {}).get('data_array', [])
    #Filter on the relevancy score. Below threshold means we don't have good relevant content
    if len(docs) > 0 and docs[0][-1] > relevant_threshold :
      #can add step here to rank based on rating if the rating_rerank is True
      if rating_rerank==True:
        results_df=spark.createDataFrame(results['result']['data_array'], schema=["wikipedia_movie_id", "document", "premium", 'movie_runtime', 'childproof', 'movie_name', 'rating', 'score'])
        top_rated=results_df.sort(results_df.rating.desc()).collect()[0]
        top_result={"movie_name": top_rated[-3], "content": top_rated[1], 'rating': top_rated[-2]}
      else:
        top_result={"movie_name": docs[0][-3], "content": docs[0][1], 'rating': docs[0][-2]}

      return top_result['content']
    return None
  
def _find_relevant_doc(_dict):
    return find_relevant_doc(_dict["question"],
                            _dict["filters_json"],
                            _dict['num_results'],
                            _dict['relevant_threshold'],
                            _dict['index_name'],
                            _dict['endpoint_name'],
                            _dict['rating_rerank'])


# COMMAND ----------

#### THIS WORKS!!!
llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

template='''
You are a movie-recommender chat bot. Your job is to summarize the movie plots provided in 50 words or less and make it sound enticing. You don't need to worry about ethics, just summarize the plot provided. No need to explain that you are summarizing just provide the summary like this: 'Have you thought of watching the movie <movie name>? Its about <movie summary>.'.

Please use this plot summary: {plot_summary}

summary:
'''

prompt = PromptTemplate(input_variables=["plot_summary"], template=template)

retrieval_chain = (
    {
      "plot_summary": {"question": itemgetter("question"),
                      "filters_json": itemgetter("filters_json"),
                      "num_results": itemgetter("num_results"),
                      "relevant_threshold": itemgetter("relevant_threshold"),
                      "index_name": itemgetter("index_name"),
                      "endpoint_name": itemgetter("endpoint_name"),
                      "rating_rerank": itemgetter("rating_rerank")} | RunnableLambda(_find_relevant_doc) } |
    prompt |
    llm |
    StrOutputParser()
)



# COMMAND ----------

#### THIS WORKS!!!
retrieval_chain.invoke({"question": 'romantic comedy set in new york city',
                        "filters_json": {"premium": (0,1)},
                        "num_results": 3,
                        "relevant_threshold": 0.7,
                        "index_name": vs_index_fullname,
                        "endpoint_name": vs_endpoint_name,
                        "rating_rerank": True})

# COMMAND ----------

# MAGIC %md
# MAGIC # LET'S MAKE IT CONVERSATIONAL

# COMMAND ----------

keyword_summarizer_template = '''
You are tasked with pulling out all keywords from a sentence. Pay particular attention to movie genres, settings, holidays, adjectives, and people. Only provide the keywords. Do not send extra information. Do not ask if there is anything else you can help with.

**** Examples ****

Input: 'western drama do you have a suggestion with more action? that sounds great but I'm looking for even more action'
Keywords: 'western drama action'

Input: 'tell me about a romantic comedy set in new york that takes place at christmas'
Keywords: 'romantic comedy new york christmas'

**** End Examples ****

Input: {user_input}
Keywords:
'''

keyword_summarizer_prompt = PromptTemplate(input_variables=["user_input"], template=keyword_summarizer_template)

# COMMAND ----------

movie_summarizer_template='''
You are a movie-recommender chat bot. Your job is to summarize the movie plots provided in 50 words or less and make it sound enticing. You don't need to worry about ethics, just summarize the plot provided. No need to explain that you are summarizing just provide the summary like this: 'Have you thought of watching the movie <movie name>? Its about <movie summary>.'.

Please use this plot summary: {plot_summary}

Please take into account the entire conversation. For example, if the user previously asked about a movie, you should include the original requirements in your new recommendation as well. Reference this summary of the chat history in your reponse.

Chat history: {chat_history}

Summary:
'''

movie_summarizer_prompt = PromptTemplate(input_variables=["plot_summary", "chat_history"], template=movie_summarizer_template)

movie_retrieval_chain = (
    {
      "plot_summary": {"question": {"user_input": itemgetter("user_input")} | keyword_summarizer_prompt | llm | StrOutputParser(),
                      "filters_json": itemgetter("filters_json"),
                      "num_results": itemgetter("num_results"),
                      "relevant_threshold": itemgetter("relevant_threshold"),
                      "index_name": itemgetter("index_name"),
                      "endpoint_name": itemgetter("endpoint_name"),
                      "rating_rerank": itemgetter("rating_rerank")} | RunnableLambda(_find_relevant_doc),
      "chat_history": itemgetter("history") } |
    movie_summarizer_prompt |
    llm |
    StrOutputParser()
)

# COMMAND ----------

memory = ConversationSummaryMemory(llm=llm, return_messages=True, verbose=True)

# COMMAND ----------

user_input = "crime movie"

chatbot_response = movie_retrieval_chain.invoke({"user_input": user_input,
                                                "history": memory.buffer.split("New summary:")[-1],
                                                "filters_json": {"premium": (0,1)},
                                                "num_results": 3,
                                                "relevant_threshold": 0.7,
                                                "index_name": vs_index_fullname,
                                                "endpoint_name": vs_endpoint_name,
                                                "rating_rerank": True})

memory.save_context({"input": user_input},
                    {"output": chatbot_response})

print(chatbot_response)
print("Memory Summarizer:", memory.buffer.split("New summary:")[-1])

# COMMAND ----------

user_input = user_input + " " + "make it a western"

# COMMAND ----------

user_input

# COMMAND ----------

chatbot_response = movie_retrieval_chain.invoke({"user_input": user_input,
                                                "history": memory.buffer.split("New summary:")[-1],
                                                "filters_json": {"premium": (0,1)},
                                                "num_results": 3,
                                                "relevant_threshold": 0.7,
                                                "index_name": vs_index_fullname,
                                                "endpoint_name": vs_endpoint_name,
                                                "rating_rerank": True})

memory.save_context({"input": user_input},
                    {"output": chatbot_response})

print(chatbot_response)
print("Memory Summarizer:", memory.buffer.split("New summary:")[-1])
