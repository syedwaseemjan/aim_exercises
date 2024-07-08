import os
import tiktoken
import requests
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
load_dotenv()

HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

DATA_DIR = "./data"
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")

# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #
### 1. LOAD FILE

file_path = os.path.join("./data", "airbnb2020ipo.pdf")
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> file_path', file_path)
response = requests.get("https://airbnb2020ipo.q4web.com/files/doc_financials/2024/q1/fdb60f7d-e616-43dc-86ef-e33d3a9bdd05.pdf", stream=True, headers={
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
})
with open(file_path,'wb') as output:
    output.write(response.content)
pdf_loader = PyMuPDFLoader(file_path)
documents = pdf_loader.load()

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10, length_function = tiktoken_len)
split_documents = text_splitter.split_documents(documents)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", len(split_documents))

### 3. LOAD HUGGINGFACE EMBEDDINGS
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# embedding_model = HuggingFaceEndpointEmbeddings(
#     model=HF_EMBED_ENDPOINT,
#     task="feature-extraction",
#     huggingfacehub_api_token=os.environ["HF_TOKEN"],
# )


os.makedirs(VECTORSTORE_DIR, exist_ok=True) 

if os.path.exists(VECTORSTORE_PATH):
    vectorstore = FAISS.load_local(
        "./data/vectorstore", 
        embedding_model, 
        allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
    )
else:
    print("Indexing Files")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    ### 4. INDEX FILES
    ### NOTE: REMEMBER TO BATCH THE DOCUMENTS WITH MAXIMUM BATCH SIZE = 32
    for i in range(0, len(split_documents), 32):
        print(f"processing batch {i}...")
        if i == 0:
            vectorstore = FAISS.from_documents(split_documents[i:i+32], embedding_model)
            continue
        vectorstore.add_documents(split_documents[i:i+32])
    vectorstore.save_local("./data/vectorstore")
    print("File saved locally...")

hf_retriever = vectorstore.as_retriever()

print("Loaded Vectorstore")

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
chat_model = HuggingFaceEndpoint(
    endpoint_url=f"{HF_LLM_ENDPOINT}",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)
# chat_model = ChatOpenAI(model="gpt-4o")


@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 
    """
    rename_dict = {
        "Assistant" : "Midterm Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")} | rag_prompt | chat_model

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()