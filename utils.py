from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import os
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub
import time

# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)
        # Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": filename.file_id, "type=": filename.type, "size": filename.size, "unique_id": unique_id},
        ))
    return docs

# Create embeddings instance
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# Function to push data to Vector Store - Pinecone here
def push_to_pinecone(api_key, project_name, index_name, embeddings, docs):
    pc = Pinecone(api_key=api_key)
    
    # Check if the index exists, and create it if it doesn't
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
    
    # Now do your stuff with pc
    LangchainPinecone.from_documents(docs, embeddings, index_name=index_name)

# Function to pull information from Vector Store - Pinecone here
def pull_from_pinecone(api_key, project_name, index_name, embeddings):
    # For some of the regions allocated in Pinecone which are on free tier, the data takes up to 10secs for it to be available for filtering
    # so I have introduced 20secs here, if it's working for you without this delay, you can remove it :)
    # https://docs.pinecone.io/docs/starter-environment

    print("20secs delay...")
    time.sleep(20)
    pc = Pinecone(api_key=api_key)
    return LangchainPinecone.from_existing_index(index_name, embeddings)

# Function to help us get relevant documents from vector store - based on user input
def similar_docs(query, k, api_key, project_name, index_name, embeddings, unique_id):
    pc = Pinecone(api_key=api_key)
    index = pull_from_pinecone(api_key, project_name, index_name, embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k), {"unique_id": unique_id})
    return similar_docs

# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary
