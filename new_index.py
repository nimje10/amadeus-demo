from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

loader = PyPDFLoader("data/Amadeus Ticket Reissue For Travel Agencies User Guide.pdf")
#pages = loader.load_and_split()

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
#embeddings = OpenAIEmbeddings(model_kwargs={'model_name': "text-embedding-ada-002"})

index_name = "amadeus-chat"

docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)






