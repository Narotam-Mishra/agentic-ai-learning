
# chatbot frontend using streamlit with streaming

import streamlit as st
from chatbot_langgraph_backend import simple_chatbot_workflow
from langchain_core.messages import HumanMessage

# define config
CONFIG = {
    'configurable':{
        'thread_id': 'thread-9'
    }
}

# session_state to keep track of message history
# session_state -> dict
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the message history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# chat input
user_input = st.chat_input('Type here')

if user_input:
    # first store the user's message to message_history
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input,
    })
    with st.chat_message('user'):
        st.text(user_input)

    # response = simple_chatbot_workflow.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in simple_chatbot_workflow.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = {'configurable':{ 'thread_id': 'thread-9'}},
                stream_mode="messages"
            )
        )
    
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message,
    })
