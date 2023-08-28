import utils
import streamlit as st
from streaming import StreamHandler

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Context aware chatbot", page_icon="⭐")
st.header('Context aware chatbot')
st.write('Enhancing Chatbot Interactions through Context Awareness')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)')

class ContextChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"
    
    @st.cache_resource
    def setup_chain(_self):
        memory = ConversationBufferMemory()
        llm = OpenAI(model_name=_self.openai_model, temperature=0, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
