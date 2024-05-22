import os
import openai
import streamlit as st
from streamlit_chat import message
from utils import get_conversation_string, query_refiner, find_match  # Assuming these are defined properly in 'utils.py'
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Streamlit UI setup
st.subheader("Demo Chatbot - Amadeus - Ticket Reissue")

# Initialize session state variables if not present
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

OPENAI_MODEL = "gpt-4-turbo"
#OPENAI_MODEL = "gpt-3.5-turbo"

#openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name=OPENAI_MODEL)

# Check if buffer memory is initialized
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Setup prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are an expert in using Amadeus product & provide very helpful support to all the travel agents who want to use Amadeus platform for managing travel related bookings.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Create a conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Containers for chat history and text input
response_container = st.container()
text_container = st.container()

with text_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Thinking..."):
            conversation_string = get_conversation_string()  # Ensure this function is handling errors properly
            refined_query = query_refiner(conversation_string, query)  # Ensure this function is handling errors properly
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            context = find_match(refined_query)  # Ensure this function is handling errors properly
            try:
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}") #Change this to query if you dont want to use the refined query
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)
            except Exception as e:
                st.error(f"Failed to generate response: {str(e)}")

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
