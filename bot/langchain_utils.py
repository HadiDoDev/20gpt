openai_api_key = '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'
from operator import itemgetter
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient,models
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.callbacks import get_openai_callback

class LANGCHAIN:
  def  __init__(self, model_name):
    self.llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name, max_tokens=1024)
    # self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "you are a helpful assistant and you always extract and return reliable answer only from your {context}.\
       write at most 100 words in your output "), ("human", "{question}")
      ])
    def connect_to_vs(collection_name):
      url="https://4b3ee481-41e3-470d-a80e-45ffb13d9c7d.us-east4-0.gcp.cloud.qdrant.io:6333"
      qdrant_api_key = 'wlxgWdvrsyuYbOQHkV3CcmnH33XFQZPxWjRXKsTAvocWouKU_uZ2jw'
      embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
      client = QdrantClient(
          url,
          api_key=qdrant_api_key, # For Qdrant Cloud, None for local instance
      )

      db  = Qdrant(
        client=client, collection_name=collection_name,
        embeddings=embeddings,
        distance_strategy= 'COSINE'
      )
      return db
    db = connect_to_vs('dini10')
    self.db = db 
    # self.chain = (RunnableParallel({'context':db.as_retriever(),"question": RunnablePassthrough()}))|prompt|self.llm
    prompt = ChatPromptTemplate.from_messages([("system", "you are a helpful assistant and you always extract and return reliable answer only from your {context}.\
    i will gave you Multiple-choice questions and just retun correct ONE, NOTHING MORE"), ("human","{question}" )])
    self.chain = (RunnableParallel(
      {"context": itemgetter("question") | db.as_retriever(),'question': RunnablePassthrough()}
      ) | prompt |self.llm)





  def __call__(self, topic, message, chatmode=''):
    if topic=='dini10':
      print(message)
      with get_openai_callback() as cost:
        response = self.chain.invoke({'question':message}).content
        in_tokens, out_tokens = cost.prompt_tokens, cost.completion_tokens

    else:
      response, in_tokens, out_tokens = None, 0, 0
    return response, in_tokens, out_tokens