# openai_api_key = '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'
openai_api_key = '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'
from operator import itemgetter
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient,models
from langchain.vectorstores import Qdrant, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    # FewShotChatMessagePromptTemplate,
    # MessagesPlaceholder,
    # SystemMessagePromptTemplate,
    # HumanMessagePromptTemplate,
)
# from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains import create_extraction_chain

from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.callbacks import get_openai_callback
# import configs
from bot import all_dbs_length

FACTOR = 1
DOLLAR = 500000
# def connect_to_vs(collection_name):
#   url="https://4b3ee481-41e3-470d-a80e-45ffb13d9c7d.us-east4-0.gcp.cloud.qdrant.io:6333"
#   qdrant_api_key = 'wlxgWdvrsyuYbOQHkV3CcmnH33XFQZPxWjRXKsTAvocWouKU_uZ2jw'
#   embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#   client = QdrantClient(
#       url,
#       api_key=qdrant_api_key, # For Qdrant Cloud, None for local instance
#   )
#   print("In connect_to_db", flush=True)
#   db  = Qdrant(
#     client=client,
#     collection_name=collection_name,
#     embeddings=embeddings,
#     distance_strategy= 'COSINE'
#   )
#   print("Created db!!!!!!", flush=True)
#   print("DB:", db.similarity_search("خدا و آخرت"), flush=True)
#   return db

def connect_to_vs(collection_name):
  path_to_vector_store = f"./vector_stors/{collection_name}" 
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
  db = FAISS.load_local(path_to_vector_store, embeddings)
  print("Created db!!!!!!", flush=True)
  print("all_dbs_length",all_dbs_length, flush=True)

  # print("DB:", db.similarity_search("اعمال ما تقدم وما تاخر را توضیح دهید"), flush=True)

  return db

class LANGCHAIN:
  def  __init__(self, model_name):
    self.llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name, max_tokens=1024)
    self.prompt_str = '''you are a helpful assistant and you always provide reliable answer only based on your {context}.
      i will gave you {question} and just retun correct answer.do not return your previouse answer, you must answer in persian language'''


  @staticmethod
  def _generate_prompt_messages(message, prompt, dialog_messages, chat_mode):
    # path_to_vector_store = f"./vector_stors/{chat_mode}_examples" 
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # vectorstore = FAISS.load_local(path_to_vector_store, embeddings)

    # example_selector = SemanticSimilarityExampleSelector(
    # vectorstore=vectorstore,
    # k=2,)
    
    # print(example_selector.select_examples({"input":message}), flush=True)
    # examples = example_selector.select_examples({"input":message})
    # to_vectorize = [" ".join(['question: \n' + example['question'], 'answer: \n' + example['answer']]) for example in examples]
    # prompt = configs.chat_modes[chat_mode]["prompt_start"]
    # prompt += 'as fewshot examples:\n'
    # for example in to_vectorize:
    #   prompt += example
    # print("Prompt Type:", type(prompt), flush=True)
    messages = [("system", prompt)]
    # for example in examples:
    #     messages.append(("human",example['question']))
    #     messages.append(("ai",example['answer']))
    for dialog_message in dialog_messages:
        messages.append(("human", dialog_message["user"]))
        # messages.append(("ai", dialog_message["bot"]))
        # messages.append({"role": "user", "content": message})
    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     # The input variables select the values to pass to the example_selector
    #     input_variables=["context", "question"],
    #     example_selector=example_selector,
    #     # Define how each example will be formatted.
    #     # In this case, each example will become 2 messages:
    #     # 1 human, and 1 AI
    #     example_prompt=ChatPromptTemplate.from_messages(messages)
    # )
    messages.append(("human", '{question}'))
    messages.append(("ai", '?'))
    return ChatPromptTemplate.from_messages(messages)
  
  @staticmethod
  def _create_chain(prompt, llm, db):
    chain = (RunnableParallel(
      {"context": itemgetter("question") | db.as_retriever(search_kwargs={"k": 5}), 'question': RunnablePassthrough()}
      ) | prompt | llm)
    return chain
  
  @staticmethod
  def _postprocess_answer(answer):
    answer = answer.strip()
    return answer

  def __call__(self, message, dialog_messages, chatmode):
    # print("In __call__, chatmode:", chatmode, flush=True)
    chat_modes_list =  ['dini10', 'jamaeshenasi11ensani', 'ravanshenasi11ensani',
                        'joghrafi12ensani', 'dini10ensani', 'amadeghi', 'jamaeshenasi10ensani',
                        'zamin11', 'dini11', 'mohitzist11', 'joghrafi10', 'joghrafi11', 'tarikh11',
                        'tarikh12ensani', 'zist10', 'zist11', 'amadeghi9', 'dini12', 'tahlil12', 'tafakor7', 'salamat12',
                        'olom7', 'khanevadeh12', 'hoviat12', 'hedieh9', 'hedieh8', 'hedieh7', 'ejtemaee9', 'ejtemaee8',
                        'dini12ensani', 'dini12', 'ejtemaee7', 'zist12', 'economy10', 'falsafeh11'
                         ]
    if chatmode in chat_modes_list:
      db = connect_to_vs(chatmode)
      prompt = self._generate_prompt_messages(message,self.prompt_str, dialog_messages, chatmode)
      # print("Prompt:", prompt, flush=True)
      chain = self._create_chain(prompt, self.llm, db)
      # print("Message:", message, type(message), flush=True)
      with get_openai_callback() as cb:
        # print("OpenAPI Callback:", flush=True)
        answer = chain.invoke({'question':message}).content
        answer = self._postprocess_answer(answer)
        # print("Get Response:", flush=True)
        n_input_tokens, n_output_tokens, cost = cb.prompt_tokens, cb.completion_tokens, cb.total_cost
    else:
      answer, n_input_tokens, n_output_tokens, cost= None, 0, 0,0
    n_first_dialog_messages_removed = 0
    return answer, n_input_tokens, n_output_tokens, n_first_dialog_messages_removed, cost* FACTOR * DOLLAR
  

  def parse_text(self, text):
    schema = {
    "properties": {
        "question": {"type": "string"},
    },
    "required": ["question"]}
    prompt = ChatPromptTemplate.from_messages([("system", """Assist me in structuring data from an OCR-processed exam paper.
    The {text} includes questions, each potentially with multiple propositions. The OCR might lack clear separation.

    1. Identify and separate each question along with all its statements, keeping all parts together.
    2. Preserve the original Persian language and question order from the OCR {text}.
    3. Be aware of sub-questions; don't treat them as standalone questions. 
    Make reasonable assumptions for clarity, and thank you for your help.
    4.always exclude 'exam header'
          """)])
    chain = create_extraction_chain(schema, self.llm, prompt)
    try:
      with get_openai_callback() as cb:
          # print("OpenAPI Callback:", flush=True)

          response = chain.invoke({"text": str(text)})['text']
          n_input_tokens, n_output_tokens, cost = cb.prompt_tokens, cb.completion_tokens, cb.total_cost
      final_answer = [item["question"] for item in response]

    except:
      final_answer = [text]
      cost = 0.001


    return final_answer, cost * FACTOR * DOLLAR
