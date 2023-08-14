#importing necessary libraries/modules, including os, openai, and several components from llama_index, 
# langchain, and chainlit. These are used to set up and manage the conversational system.

import os
import openai

#RetrieverQueryEngine: plug retriever into the query engine. related to the query engine used to retrieve information from the vector index. 
#CallbackManager: This class manages callbacks or event handlers for actions with the query engine. 

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager


from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import chainlit as cl


openai.api_key = os.environ.get("OPENAI_API_KEY")

#building the storage context and loading the index.

# It first tries to load an index from storage using the provided storage context. 
# If successful, it stores the loaded index in the variable index.

#In the except block, the code imports the necessary components for creating an index from scratch.
try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.on_chat_start
async def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0, #deterministic output
            model_name="gpt-3.5-turbo",
            streaming=True,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=True,
    )

    cl.user_session.set("query_engine", query_engine)

#decorator is used to indicate that the following function should be called whenever a new message is received

@cl.on_message
async def main(message): #This defines an asynchronous function named main that takes a single parameter, message.from the user.
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine retrieves the query engine instance from the user session.
    response = await cl.make_async(query_engine.query)(message) #triggers the query engine to process the user's message and generate a response.

    response_message = cl.Message(content="")  #hold the response message. The initial content of the response is set to an empty string.

    for token in response.response_gen: #This loop iterates over the tokens generated in the response.response_gen generator
        await response_message.stream_token(token=token) #Within the loop, each token is streamed to the response_message. Build response by adding tokens gradually. 

    if response.response_txt: # if the response.response_txt attribute exists, indicating that the response contains text.
        response_message.content = response.response_txt  #the text is assigned to the content attribute of the response_message.

    await response_message.send() #Finally, the response_message is sent
