openai_api_key = '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'
# openai_api_key =  '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'
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

class LANGCHAIN:
  def  __init__(self, model_name):
    self.llm = ChatOpenAI(api_key=openai_api_key, model=model_name, max_tokens=1024)
    self.qd_url = '"https://4b3ee481-41e3-470d-a80e-45ffb13d9c7d.us-east4-0.gcp.cloud.qdrant.io:6333"'
    self.qdrant_api_key = 'wlxgWdvrsyuYbOQHkV3CcmnH33XFQZPxWjRXKsTAvocWouKU_uZ2jw'
    self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([("system", "you are a helpful assistant and\
                                                 you always extract and return reliable answer only from your {context}.\
                                                 write at most 100 words in your output "),
                                                   ("human","{question}" )])
    db = self.connect_to_vs('dini10', self.qd_url, self.qdrant_api_key, self.embeddings)
    self.chain = ({'context':db.as_retriever(),"question": RunnablePassthrough()})|prompt|self.llm

  @staticmethod
  def connect_to_vs(collection_name, url, qdrant_api_key, embeddings):
    client = QdrantClient(
        url,
        api_key=qdrant_api_key, # For Qdrant Cloud, None for local instance
    )

    db  = Qdrant(
        client=client, collection_name=collection_name,
        embeddings=embeddings,
        distance_strategy= 'COSINE'
    ) #"andishe_1"
    return db



  def __call__(self, topic, message, chatmode=''):
    if topic=='dini10':
      response = self.chain.invoke({'question':message}).content
    else:
      response = None
    #   response = self._aris_qa(self.llm, encoded_image, question)
    # if topic=='ariss_detection':
    #   response = self._aris_detector(self.llm, encoded_image)
    # if topic=='aipet_detection':
    #   response = self._aipet_detector(self.llm, encoded_image)
    return response
