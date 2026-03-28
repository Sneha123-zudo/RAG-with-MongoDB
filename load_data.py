from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import key_param


# -------------------------------
# MongoDB Connection
# -------------------------------
client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]


# -------------------------------
# Load PDF
# -------------------------------
loader = PyPDFLoader(r"C:\Users\admin\Desktop\MongoDB-RAG\sample_files\mongodb.pdf")
pages = loader.load()

cleaned_pages = []
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)


# -------------------------------
# Text Splitting
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150
)

split_docs = text_splitter.split_documents(cleaned_pages)


# -------------------------------
# Custom Embedding (Sentence Transformers)
# -------------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


embeddings = SentenceTransformerEmbeddings()


# -------------------------------
# Store in MongoDB Vector Store
# -------------------------------
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs,
    embeddings,
    collection=collection
)

print("Data successfully stored with embeddings 🚀")
