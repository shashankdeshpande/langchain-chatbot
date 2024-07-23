import os
import utils
import requests
import traceback
import validators
import streamlit as st
from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

st.set_page_config(page_title="ChatWebsite", page_icon="🔗")
st.header('Chat with Website')
st.write('Enable the chatbot to interact with website contents.')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/6_%F0%9F%94%97_chat_with_website.py)')

class ChatbotWeb:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()

    def scrape_website(self, url):
        content = ""
        try:
            base_url = "https://r.jina.ai/"
            final_url = base_url + url
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'
                }
            response = requests.get(final_url, headers=headers)
            content = response.text
        except Exception as e:
            traceback.print_exc()
        return content

    # @st.cache_resource(show_spinner='Analyzing webpage', ttl=3600)
    def setup_vectordb(_self, websites):
        # Scrape and load documents
        docs = []
        for url in websites:
            docs.append(Document(
                page_content=_self.scrape_website(url),
                metadata={"source":url}
                )
            )

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
        return vectordb

    def setup_qa_chain(self, vectordb):

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
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
            verbose=True
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        # User Inputs
        if "websites" not in st.session_state:
            st.session_state["websites"] = []

        web_url = st.sidebar.text_area(
            label='Enter Website URL',
            placeholder="https://",
            help="To add another website, modify this field after adding the website."
            )
        if st.sidebar.button(":heavy_plus_sign: Add Website"):
            valid_url = web_url.startswith('http') and validators.url(web_url)
            if not valid_url :
                st.sidebar.error("Invalid URL! Please check website url that you have entered.", icon="⚠️")
            else:
                st.session_state["websites"].append(web_url)

        if st.sidebar.button("Clear", type="primary"):
            st.session_state["websites"] = []
        
        websites = list(set(st.session_state["websites"]))

        if not websites:
            st.error("Please enter website url to continue!")
            st.stop()
        else:
            st.sidebar.info("Websites - \n - {}".format('\n - '.join(websites)))

            vectordb = self.setup_vectordb(websites)
            qa_chain = self.setup_qa_chain(vectordb)

            user_query = st.chat_input(placeholder="Ask me anything!")
            if websites and user_query:

                utils.display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    result = qa_chain.invoke(
                        {"question":user_query},
                        {"callbacks": [st_cb]}
                    )
                    response = result["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # to show references
                    for idx, doc in enumerate(result['source_documents'],1):
                        url = os.path.basename(doc.metadata['source'])
                        ref_title = f":blue[Reference {idx}: *{url}*]"
                        with st.popover(ref_title):
                            st.caption(doc.page_content)

if __name__ == "__main__":
    obj = ChatbotWeb()
    obj.main()
