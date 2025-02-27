import json
import streamlit as st
from streaming import StreamHandler
import utils
import chromadb
import ollama
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



def get_ollama_embedding(text):
    """Generate embeddings using Ollama."""
    response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
    return response['embedding']


def normalize_embedding(embedding):
    """Apply L2 normalization to an embedding."""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


class RAG:
    
    def __init__(self, chromadb_path):
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_collection(name="embeddings")

    def retrieve_top_k(self, query_text, top_k=5):
        query_embedding = get_ollama_embedding(query_text)
        query_embedding = normalize_embedding(query_embedding)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0] if "documents" in results else []


class EcommerceAgent:
    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.name = "EcommerceAgent"
        #self.rag = RAG(chromadb_path="chromadb_store")
    
    @st.cache_resource()
    def setup_vectordb(_self,):
        embeddings = OllamaEmbeddings(model="mxbai-embed-large",)
        chroma = Chroma(
            persist_directory="chromadb_store",
            collection_name="embeddings",
            embedding_function=embeddings
        )
        return chroma

    def setup_chain(self, vectordb):
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        return qa_chain
    
    @utils.enable_chat_history
    def main(self):
        vectordb = self.setup_vectordb()
        chain = self.setup_chain(vectordb)
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"question":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(EcommerceAgent, user_query, response)
                
                # to show references
                for idx, doc in enumerate(result['source_documents'],1):
                    item = json.loads(doc.page_content)
                    item_id = item['id']
                    item_name = item['name']
                    item_buy_url = item['buy_url']
                    with st.popover(item_name):
                        st.caption(f"Item ID: {item_id}, Buy URL: {item_buy_url}")

if __name__ == "__main__":
    obj = EcommerceAgent()
    obj.main()