
# chatbot frontend using streamlit with streaming

import streamlit as st
from chatbot_backend_with_db import simple_chatbot_workflow, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# ********************************* utility functions ************************** #

# dynamically generate thread_id
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

# reset chat
def reset_chat():
    # generate new thread id and add it to current session
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id

    # add thread on click of reset
    add_thread(st.session_state['thread_id'])

    # reset message history (empty it)
    st.session_state['message_history'] = []

# add new chat thread on click of new chat
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

# return all messages from a thread_id
def load_conversation(thread_id):
    CONFIG = {
        'configurable':{
            'thread_id': thread_id,
        }
    }
    state = simple_chatbot_workflow.get_state(config=CONFIG)
    return state.values.get('messages', [])


# ************************ Session Setup ********************************** #

# session_state to keep track of message history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# create a list to store all thread_ids
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# add thread to current session state
add_thread(st.session_state['thread_id'])

# ************************** Sidebar UI ************************ #

# add sidebar
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

# display list all of thread_ids
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        # fix message format issue
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({
                'role': role,
                'content': msg.content
            })
        
        st.session_state['message_history'] = temp_messages

# ****************************** Main UI **************************** #

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

    # define config
    CONFIG = {
        'configurable':{
            'thread_id': st.session_state['thread_id']
        }
    }

    # generate ai message
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in simple_chatbot_workflow.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode="messages"
            )
        )
    
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message,
    })
