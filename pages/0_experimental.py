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


class OllamaEmbeddingsL2(OllamaEmbeddings):
    """Ollama embedding class with L2 normalization to the ollama embeddings"""
    
    def _normalize_embedding(self, embedding):
        """Normalize embedding using L2 normalization."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def embed_documents(self, texts):
        """Embed a batch of documents with L2 normalization."""
        embeddings = super().embed_documents(texts)
        return [self._normalize_embedding(embedding) for embedding in embeddings]

    def embed_query(self, text):
        """Embed a query with L2 normalization."""
        embedding = super().embed_query(text)
        return self._normalize_embedding(embedding)


class EcommerceAgent:
    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.name = "EcommerceAgent"
        #self.rag = RAG(chromadb_path="chromadb_store")
    
    @st.cache_resource()
    def setup_vectordb(_self,):
        embeddings = OllamaEmbeddingsL2(model="mxbai-embed-large",)
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