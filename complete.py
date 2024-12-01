#%%
# Import Libraries
import streamlit as st
import os
from dotenv import load_dotenv
from docx import Document
import openai
import pandas as pd
import re
import time
import pickle
import importlib
import configparser
import tiktoken

# Custom Functions Module
import to_pager_functions_2 as tp
importlib.reload(tp)

import pdf_chat_functions as pc
importlib.reload(pc)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OpenAI_key")

# Check API key
if openai.api_key is None:
    st.error("Error: OpenAI API key not found. Make sure it is set in environment variables or Streamlit secrets.")
    st.stop()

# Set Page Configuration
st.set_page_config(page_title='AI Gradiente', page_icon=':robot:')
#st.subheader('SUCCESS COMES WHEN PREPARATION MEETS OPPORTUNITY')

# Add custom font and styles
st.markdown("""
    <style>

    /* Apply a generic font globally */
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }   
    
    /* Optional: Customize specific elements */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 500;
        color: #003866;  /* Adjust header color if needed */
    }
    .stButton>button {
        font-family: Arial, sans-serif;
        font-weight: 700;
        color: white;
        background-color: #E41A13;  /* Button background color */
        border-radius: 5px;
        border: none;
    }
    .stMarkdown {
        color: #003866;  /* Paragraph text color */
    }
    </style>
""", unsafe_allow_html=True)

# Display Banner Image
banner_path = "AI GRADIENTE VETTORIALE_page-0001.jpg"  # Update with the correct path
st.image(banner_path, use_container_width=True)

st.markdown("<h3 style='font-size:25px;'>Select your application:</h3>", unsafe_allow_html=True)

# Inject custom CSS to reduce the margin above the select box
st.markdown(
    """
    <style>
    div[data-testid="stSelectbox"] {
        margin-top: -50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

option = st.selectbox(
    '',  # Leave label empty because it's already displayed above
    ('Select an application', 'Chatbot with PDFs', '2Pager Generator')
)

# Chatbot Functionality
def chatbot_with_pdfs(default=True, pdf_docs=None):

    if default:
        st.header('Chat with multiple PDFs :books:')

    # Initialize Session State
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if default:
        # Existing code for default behavior
        with st.sidebar:
            st.subheader('Your documents')
            pdf_docs = st.file_uploader('Upload your PDFs here and click on Process', 
                                        accept_multiple_files=True)
            if st.button('Process'):
                if pdf_docs:
                    with st.spinner('Processing'):
                        # Process PDFs
                        raw_text = pc.get_pdf_text(pdf_docs)
                        text_chunks = pc.get_text_chunks(raw_text)
                        vectorstore = pc.get_vectorstore(text_chunks)
                        st.session_state.conversation = pc.get_conversation_chain(vectorstore)
                        st.session_state.chat_history = []
                        st.success('Processing complete! You can now ask questions.')
                else:
                    st.warning('Please upload at least one PDF file before processing.')
    else:
            if pdf_docs:
                if st.session_state.conversation is None:
                    with st.spinner('Processing'):
                        
                        raw_text = pc.get_text_from_files(pdf_docs)
                        text_chunks = pc.get_text_chunks(raw_text)
                        vectorstore = pc.get_vectorstore(text_chunks)
                        st.session_state.conversation = pc.get_conversation_chain(vectorstore)
                        st.session_state.chat_history = []
                        st.success('Processing complete! You can now ask questions.')
            else:
                st.error('No documents to process. Please provide PDFs.')

    # Input for questions
    user_question = st.chat_input('Ask a question about your documents:')

    # Process the question
    if user_question and st.session_state.conversation:
        with st.spinner("Fetching response..."):
            try:
                # Get the response from the conversation chain
                response = st.session_state.conversation({'question': user_question})
                answer = response['answer']  # Assuming response contains an 'answer' key

                #Update chat history in session state
                st.session_state.chat_history.append({'question': user_question, 'answer': answer})
                #Refresh UI to display the updated chat history

            except Exception as e:
                st.error(f"Error: {e}")

    # Display chat history with images
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            # User's question
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; border: 1px solid #d6d6d6; border-radius: 25px; padding: 10px; margin-bottom: 10px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/1077/1077012.png" alt="user" width="30" style="vertical-align: middle; margin-right: 10px;">
                    <b>You:</b> {chat['question']}
                </div>
                """,
                unsafe_allow_html=True
            )
            # Chatbot's response
            st.markdown(
                f"""
                    <b>AI Gradiente:</b> {chat['answer']}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Spacer to push the input box to the bottom
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)


# Document Generator Functionality
def document_generator():
    
    milestone = [1]
    steps = 5

    # Preloaded Files
    xlsx_file = "prompt_db.xlsx"
    docx_file = "to_pager_template.docx"

    doc_copy = Document(docx_file)
    
    # Initialize the OpenAI client
    client = openai.OpenAI()

    
    # Create a ConfigParser instance
    config = configparser.ConfigParser()
    
    # Read the .cfg file
    config.read('assistant_config.cfg')  # Replace with your file path

    st.header('2Pager Generator :page_facing_up:')
    
    # Inputs or configurations for the document generator
    st.markdown('Upload your files here:')

    st.markdown(
    """
    <style>
    div[data-testid="stFileUploader"] {
        margin-top: -50px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


    # Template Path Input
    pdf_docs = st.file_uploader('',accept_multiple_files=True)
    #st.write(f'{type(pdf_docs)}')
    
    st.markdown('Project title:')

    hide_enter_message = (
    """
    <style>
    div[data-testid="stTextInput"] {
        margin-top: -50px;
    }
    div[data-testid="InputInstructions"] > span:nth-child(1) {
    visibility: hidden;
    }
    </style>
    """   )
    st.markdown(hide_enter_message, unsafe_allow_html=True)
    project_title = st.text_input("")

    gen_button = st.button('Generate Document')

    # Start the generation process
    if gen_button:
        
        #Persistent variables that we need across sessions
        file_streams = pdf_docs
        output_path = f'{project_title}.docx'

        #This willl update the session_state so that 
        st.session_state.document_generated = True
        st.session_state.generated_doc_path = output_path
        st.session_state.file_streams = file_streams
        st.session_state.project_title = project_title
    
        #Initialize progress bar and creating a placeholder for dynamic text
        progress_bar = st.progress(0)  
        message_placeholder = st.empty() 

        tp.update_progressbar(progress_bar, message_placeholder,
                              milestone, steps,
                              message="Learning from the presentation...")
                
        # Initialize variables
        temp_responses = []
        answers_dict = {}

        configuration = tp.assistant_config(config, 'BO')

        assistant_identifier = tp.create_assistant(client, 'final_test', configuration)

        vector_store = client.beta.vector_stores.create(name="Business Overview")
        vector_store_id = vector_store.id
        
        tp.load_file_to_assistant(client, vector_store_id,
                                    assistant_identifier, file_streams)

        
        # Retrieve prompts and formatting requirements
        try:
            prompt_list, additional_formatting_requirements, prompt_df = tp.prompts_retriever(
                'prompt_db.xlsx', ['BO_Prompts', 'BO_Format_add'])
        except Exception as e:
            st.error(f"Error retrieving prompts: {e}")
            return

        tp.update_progressbar(progress_bar, message_placeholder,
                              milestone, steps,
                              message="Generating Business Overview...")
        
        for prompt_name, prompt_message in prompt_list:
            prompt_message_f = tp.prompt_creator(prompt_df, prompt_name, 
                                                prompt_message, additional_formatting_requirements,
                                                answers_dict)
            
            assistant_response, thread_id = tp.separate_thread_answers(openai, prompt_message_f, 
                                                            assistant_identifier)
            
            assistant_response = tp.warning_check(assistant_response, client,
                                                  thread_id, prompt_message, 
                                                  assistant_identifier)
            
            if assistant_response:
                temp_responses.append(assistant_response)
                assistant_response = tp.remove_source_patterns(assistant_response)
                answers_dict[prompt_name] = assistant_response
                tp.document_filler(doc_copy, prompt_name, assistant_response)
            else:
                st.warning(f"No response for prompt '{prompt_name}'.")
        
        
        #REFERENCE MARKET CREATION
        
        configuration = tp.assistant_config(config, 'RM')
        assistant_identifier = tp.create_assistant(client, 'final_test', configuration)

        vector_store = client.beta.vector_stores.create(name="Reference Market")
        vector_store_id = vector_store.id
        
        tp.load_file_to_assistant(client, vector_store_id,
                                    assistant_identifier, file_streams)
        

        tp.update_progressbar(progress_bar, message_placeholder,
                              milestone, steps,
                              message="Searching online...")
        
        retrieved_files = tp.html_retriever(file_streams)
        st.session_state.retrieved_files = retrieved_files

        all_files = file_streams + retrieved_files
        st.session_state.all_files = all_files

        if retrieved_files:

            tp.load_file_to_assistant(client, vector_store_id,
                                        assistant_identifier, retrieved_files,
                                        uploaded = False)



        tp.update_progressbar(progress_bar, message_placeholder,
                              milestone, steps,
                              message="Generating Market Analysis...")
        
        prompt_list, additional_formatting_requirements, prompt_df = tp.prompts_retriever('prompt_db.xlsx', 
                                                                                        ['RM_Prompts', 'RM_Format_add'])
        for prompt_name, prompt_message in prompt_list:

            prompt_message_f = tp.prompt_creator(prompt_df, prompt_name, 
                                            prompt_message, additional_formatting_requirements,
                                            answers_dict)

            assistant_response, thread_id = tp.separate_thread_answers(openai, prompt_message_f, 
                                                            assistant_identifier)
            
            assistant_response = tp.warning_check(assistant_response, client,
                                                  thread_id, prompt_message, 
                                                  assistant_identifier)
            

            if assistant_response:
                print(f"Assistant response for prompt '{prompt_name}': {assistant_response}")

            temp_responses.append(assistant_response)

            assistant_response = tp.remove_source_patterns(assistant_response)

            answers_dict[prompt_name] = assistant_response

            tp.document_filler(doc_copy, prompt_name, assistant_response)
    

        tp.update_progressbar(progress_bar, message_placeholder,
                              milestone, steps,
                              message = "Formatting the document...")

        tp.adding_headers(doc_copy, project_title)

    if st.session_state.get('document_generated', False):
        output_path = st.session_state.generated_doc_path
        doc_copy.save(output_path)
        with open(output_path, "rb") as doc_file:
            btn = st.download_button(
                label="Download Document",
                data=doc_file,
                file_name=output_path,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        fact_check_button = st.button('Fact Check')
        if fact_check_button:
            st.session_state.fact_check = True

    if st.session_state.get('fact_check', False):

        chatbot_with_pdfs(default=False, pdf_docs=st.session_state.all_files)

# Main Function
def main():
    if option == 'Chatbot with PDFs':
        chatbot_with_pdfs()
    elif option == '2Pager Generator':
        document_generator()
    else:
        pass

if __name__ == '__main__':
    main()
