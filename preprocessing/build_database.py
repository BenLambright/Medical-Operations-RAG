# followed along with this page: https://python.langchain.com/docs/tutorials/rag/
# followed this documentation specifically as well: https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
import pickle

# importing the documents from our data preprocessing:
from preprocessing import documents

# initiate vector store, from documentation
# I used an open source model recommended from one of the tutorials, but I also tried OpenAI (which ran much faster but costs money)
# It isn't a lot of money (a few cents), so I might compare these two embeddings when actually running experiments
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# adding our documents to the vector database
print("processing data...")
document_ids = vector_store.add_documents(documents)
print("document_ids completed")

# store the data, both the vector store and the document ids because we might need both
# really the document store is what is expensive though
with open("../vector_store.pkl", "wb") as f:
    pickle.dump(vector_store, f)
with open("../document_ids.pkl", "wb") as f:
    pickle.dump(document_ids, f)



        ### The following code is from LangChain tutorials that I also used for sanity checks on my datasest  ###
# # test: adding documents
# document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
# document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
# document_3 = Document(id="3", page_content="i will be deleted :(")
#
# documents = [document_1, document_2, document_3]
# vector_store.add_documents(documents=documents)
#
# # documentation example: inspecting documents
# top_n = 10
# for index, (id, doc) in enumerate(vector_store.store.items()):
#     if index < top_n:
#         # docs have keys 'id', 'vector', 'text', 'metadata'
#         print(f"{id}: {doc['text']}")
#     else:
#         break
#
# documentation example: search documents
results = vector_store.similarity_search(query="Surgery",k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")