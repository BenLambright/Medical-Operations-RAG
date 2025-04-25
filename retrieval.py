import os
os.environ["TOGETHER_API_KEY"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]    = ""
print("config completed")

from langchain import hub
import pickle
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_together.chat_models import ChatTogether
from langgraph.graph import START, StateGraph


PATH = "vector_store.pkl"


def get_data(path):
    with open(path, "rb") as f:
        vector_store = pickle.load(f)

    return vector_store


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class Retriever:
    def __init__(self, data_path=PATH):
        self.llm = ChatTogether(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",  # or whichever chat model you prefer
            max_tokens=512,                   # required by the completions API :contentReference[oaicite:0]{index=0}
            temperature=0.0,
            verbose=True,
        )
        self.vector_store = get_data(data_path)
        self.prompt = hub.pull("rlm/rag-prompt")

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def build_retriever_graph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph


test = Retriever()
retriever = test.build_retriever_graph()

# result = retriever.invoke({"question": "How much anesthesia would I use to for the following operation: oungswick osteotomy with internal screw fixation of the first right metatarsophalangeal joint of the right foot."})
result = retriever.invoke({"question":  "Given the following information, describe the operation (including materials) in detail. A man in his mid-sixties was diagnosed with a lateral renal tumor in the lower pole of the left kidney. Imaging and biopsy confirmed renal cell carcinoma."})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')