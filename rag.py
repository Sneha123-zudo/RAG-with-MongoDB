from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer
import ollama
import key_param


# -------------------------------
# Custom Embeddings
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
# MongoDB Vector Store
# -------------------------------
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGODB_URI,
    dbName + "." + collectionName,
    embeddings,
    index_name=index,
)


# -------------------------------
# Query Function
# -------------------------------
def query_data(query):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    template = """
Use the following context to answer the question.
If answer not in context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Retrieve + format context
    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }

    response_parser = StrOutputParser()

    rag_chain = retrieve | custom_rag_prompt
    prompt_value = rag_chain.invoke(query)

    # Step 1: Get prompt
    final_prompt = prompt_value.to_string()

    # Step 2: Send to Ollama
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return response["message"]["content"]


# -------------------------------
# Run
# -------------------------------
print(query_data("When did MongoDB begin supporting multi-document transactions?"))

