import streamlit as st

st.set_page_config(
    page_title="Langchain Chatbot",
    page_icon='💬',
    layout='wide'
)

st.header("Chatbot Implementations with Langchain")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Flangchain-chatbot.streamlit.app&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
""")
st.write("""
Langchain is a powerful framework designed to streamline the development of applications using Language Models (LLMs). It provides a comprehensive integration of various components, simplifying the process of assembling them to create robust applications.

Leveraging the power of Langchain, the creation of chatbots becomes effortless. Here are a few examples of chatbot implementations catering to different use cases:

- **Basic Chatbot**: Engage in interactive conversations with the LLM.
- **Context aware chatbot**: A chatbot that remembers previous conversations and provides responses accordingly.
- **Chatbot with Internet Access**: An internet-enabled chatbot capable of answering user queries about recent events.
- **Chat with your documents** Empower the chatbot with the ability to access custom documents, enabling it to provide answers to user queries based on the referenced information.

To explore sample usage of each chatbot, please navigate to the corresponding chatbot section.
""")