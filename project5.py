import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler

# Setup
MODEL = "gpt-4o-mini"
db_name = "vector_db"
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Load documents
folders = glob.glob("knowledge-base/*")
documents = []
text_loader_kwargs = {'encoding': 'utf-8'}

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embeddings and Vector Store
embeddings = OpenAIEmbeddings()
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# Inspect Vectorstore
collection = vectorstore._collection
vectors = np.array(collection.get(include=["embeddings"])["embeddings"])
doc_data = collection.get(include=['documents', 'metadatas'])
documents = doc_data['documents']
metadatas = doc_data['metadatas']
doc_types = [meta['doc_type'] for meta in metadatas]

# Color mapping
known_types = ['products', 'employees', 'contracts', 'company']
color_map = dict(zip(known_types, ['blue', 'green', 'red', 'orange']))
colors = [color_map.get(t, 'gray') for t in doc_types]

# 2D t-SNE Visualization
tsne_2d = TSNE(n_components=2, random_state=42)
reduced_2d = tsne_2d.fit_transform(vectors)
fig_2d = go.Figure(data=[go.Scatter(
    x=reduced_2d[:, 0], y=reduced_2d[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"{t}: {d[:100]}" for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])
fig_2d.update_layout(title="2D Chroma Vector Store Visualization", width=800, height=600)
fig_2d.show()

# 3D t-SNE Visualization
tsne_3d = TSNE(n_components=3, random_state=42)
reduced_3d = tsne_3d.fit_transform(vectors)
fig_3d = go.Figure(data=[go.Scatter3d(
    x=reduced_3d[:, 0], y=reduced_3d[:, 1], z=reduced_3d[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"{t}: {d[:100]}" for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])
fig_3d.update_layout(title="3D Chroma Vector Store Visualization", width=900, height=700)
fig_3d.show()

# Conversation setup
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()]
)

# Gradio Chat
def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
