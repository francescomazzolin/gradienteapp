#%%

"""
Packages for environment selection 
"""

import os  # Missing import for 'os'
from dotenv import find_dotenv, load_dotenv

"""
Packages for document writing
"""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
#from docx.shared import Inches

import openai

import pandas as pd
import re
import time
import pickle
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from datetime import datetime


import streamlit as st


def get_pdf_files_in_directory(directory):
    """Returns a list of PDF files in the given directory."""
    return [file for file in os.listdir(directory) if file.endswith('.pdf')]


"""
==================================================================================================================
Assistant Creator and manager functions
==================================================================================================================
"""


def assistant_config(config, qualifier):

    res = {}

    model = config.get(f'assistant_{qualifier}', 'model', fallback=None)
    instructions = config.get(f'assistant_{qualifier}', 'instruction', fallback=None)
    temperature = config.getfloat(f'assistant_{qualifier}', 'temperature', fallback=None)
    topP = config.getfloat(f'assistant_{qualifier}', 'topP', fallback=None)

    res['model'] = model
    res['instructions'] = instructions
    res['temperature'] = temperature
    res['topP'] = topP
    return res 

def create_assistant(client, name, config):
    instructions = config['instructions']
    model = config['model']
    temp = config['temperature']
    topP = config['topP']

    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=[{"type": "file_search"}],
        model=model,
        temperature= temp,
        top_p= topP

    )
    return assistant.id  # Return the assistant ID

def load_file_to_assistant(client, vector_storeid ,
                           assistant_identifier, pdf_docs,
                           uploaded = True):

    # Get the current directory
    #current_directory = os.getcwd()

    # Get a list of PDF files in the current directory
    #pdf_files = get_pdf_files_in_directory(current_directory)

    #vector_store = client.beta.vector_stores.create(name="Business Overview")

    #pdf_dirs = [pdf._file_urls.upload_url for pdf in pdf_docs]
    
    #file_streams = [open(path, "rb") for path in pdf_files]
    #file_streams = [open(path, "rb") for path in pdf_dirs]

    if uploaded: 
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id= vector_storeid, files=pdf_docs
        )

        print(file_batch.status)
        print(file_batch.file_counts)

    else:

        # Open each file in binary mode
        file_streams = [open(file_path, "rb") for file_path in pdf_docs]

        try:
            # Upload the files to the vector store
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id= vector_storeid, files=file_streams
            )

            print(file_batch.status)
            print(file_batch.file_counts)


        finally:
            # Ensure all file streams are closed
            for stream in file_streams:
                stream.close()


    assistant = client.beta.assistants.update(
    assistant_id= assistant_identifier,
    tool_resources={"file_search": {"vector_store_ids": [vector_storeid]}},
    )



"""
==================================================================================================================
Assistant Question and Answering functions
==================================================================================================================
"""


def get_answer(client, run, thread_identifier):
    while not run.status == "completed":
        #print("Waiting for answer...")
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_identifier,
            run_id=run.id
        )

"""
The following function is about loading the prompts we will use to fill the document.

This retrieves, from a .xlsx file, both the prompt and the placeholder metadata.

The placeholders corresponds to the ones in the .docx document and will be used 
select the appropriate place in which the assistant answer will be placed in 
the final document.
"""

def prompts_retriever(file_name, sheet_list):

    prompt_sheet = sheet_list[0]
    formatting_sheet = sheet_list[1]
        
    prompt_df = pd.read_excel(
        file_name, 
        sheet_name=prompt_sheet, 
        keep_default_na=True,  # Keep pandas default missing value recognition
        na_values=['']         # Treat empty strings as NaN
    )
    prompt_list = list(zip(prompt_df['Placeholder'], prompt_df['Prompt']))

    temp_df = pd.read_excel(file_name,sheet_name=formatting_sheet)

    additional_formatting_requirements = temp_df.iloc[0,0]

    return prompt_list, additional_formatting_requirements, prompt_df


def prompt_creator(prompt_df, prompt_name, 
                   prompt_message, additional_formatting_requirements,
                   answers_dict):
    
    
    print(prompt_message)

    row = prompt_df[prompt_df['Placeholder'] == prompt_name]

    if pd.isna(row['Links'].iloc[0]):

        prompt_message_format = prompt_message + additional_formatting_requirements

    else:

        reference = answers_dict[row['Links'].iloc[0]]
        prompt_message_format = reference
        prompt_message_format += prompt_message + additional_formatting_requirements
    
    """
    We will iterate through all the prompts that are present in the .xlsx file.

    The prompt_list object is a list of tuples with:

        1) prompt_name = The placeholder in the .docx file that is associated with the 
        current prompt.

        2) prompt_message = The prompt itself that will be used to ask the assistant a question.
    """
    
    print(prompt_message_format)
    
    return prompt_message_format
    



def separate_thread_answers(client, prompt_message_format,
                            assistant_identifier, 
                            same_chat = False, thread_id = ''):
    if not same_chat: 

        thread = client.beta.threads.create()
        thread_identifier = thread.id

    else:

        thread_identifier = thread_id

    """
    We essentially append our message to the current thread, to query the assistant
    """

    user_message = client.beta.threads.messages.create(
        thread_id=thread_identifier,
        role="user",
        content=prompt_message_format
    )

    """
    This is the actual interaction with the OpenAI assistant
    """

    run = client.beta.threads.runs.create(
        thread_id=thread_identifier,
        assistant_id= assistant_identifier  
    )

    """
    In order to achieve a sequential workflow in which we move to the next prompt 
    only when the previous one was answered, we added this while loop to prevent
    moving forward until prompt completion.

    """
    
    run = get_answer(client, run, thread_identifier)
  
    
    """
    We retrieve the entire list of messages that are part of the thread.

    By looping through the data attribute we are moving from the last message 
    upwards to the first.

!!!!!!!!!!!!!!!!!!!
    We will retrieve the first message that the answer from the assistant
    whose content is textual.
!!!!!!!!!!!!!!!!!!!
    """

    messages = client.beta.threads.messages.list(thread_id=thread_identifier)
    assistant_response = None

    for message in messages.data:  
        if message.role == "assistant":
            for content_block in message.content:
                if content_block.type == "text":
                    assistant_response = content_block.text.value
                    break
            if assistant_response:
                break

    return assistant_response, thread_identifier


def missing_warning(client, thread_id, prompt, assistant_identifier):

    question = """Given that, with the information in the files uploaded to the assistant, the model was not able to answer the following question:\n"""
    question = question + prompt

    question = """
    Please write what will be a warning to the user that the model was not able to find the answer.

    It should follow: "The AI Assistant did not find/was not confident enough to write about: {the theme of the question}
    """

    warning, x = separate_thread_answers(client, prompt, assistant_identifier)

    warning += " Highlight!$%"

    return warning

def warning_check(answer, client, thread_id, prompt, assistant_identifier):

    if "not_found" not in answer.lower():

        return answer
    
    else:

        warning = missing_warning(client, thread_id, prompt, assistant_identifier)
        st.write(f'To the prompt: {prompt}')
        st.write(f'Gives waring: {warning}')

        return warning


"""
==================================================================================================================
REFERENCE MARKET PART
==================================================================================================================
"""

def get_pdf_text(pdf_docs): 
    """Extracts text from a list of PDF files."""
    text = "" 
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf) 

        for page in pdf_reader.pages: 
            text += page.extract_text() or "" 
                
    return text

def html_retriever(uploaded_files):
    #st.write(f"{uploaded_files}")

    html_dir = "retrieved_html_files"
    os.makedirs(html_dir, exist_ok=True)

    extracted_text = get_pdf_text(uploaded_files)
    #st.write(f"{extracted_text}")

    """
    Finding the URLs inside the files
    """
        
    url_df = pd.DataFrame(columns = ['single_line', 'double_line', 'triple_line'])
    url_pattern = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    lines = extracted_text.split("\n")

    found = []

    for idx, line in enumerate(lines):

        if re.search(r"https?://", line):

            l = []
            s = re.findall(url_pattern, line)
            l.append(s[0])

            l.append(s[0] + lines[idx + 1])
            l.append(s[0] + "".join(lines[idx+1:idx+3]))

            url_df.loc[len(url_df)] = l

    
        
    # Ensure the output directory exists
    #os.makedirs('html_files', exist_ok=True)

    # List to hold file paths
    html_file_paths = []

    # Download HTML content for each URL
    for idx in url_df.index:
        for column in url_df.columns:
            url = url_df.loc[idx, column]
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check for HTTP errors

                # Save HTML content to a file
                file_name = f"page_{idx}.html"
                file_path = os.path.join(html_dir, file_name)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(response.text)

                html_file_paths.append(file_path)
                #st.write(f"HTML content retrieved and saved from {url}")
                break  # Stop trying other columns once successful
            except Exception as e:
                #st.warning(f"Failed to fetch {url}: {e}")
                continue

    return html_file_paths

    
"""
==================================================================================================================
App design functions
==================================================================================================================
"""

def update_progressbar(progress_bar, message_placeholder,
                       milestone, steps):
    progress_bar.progress(milestone / steps)
    message_placeholder.markdown("Formatting the document...")
    time.sleep(1)  
    milestone += 1


"""
==================================================================================================================
DOCUMENT FORMATTING
==================================================================================================================
"""

def remove_source_patterns(text):
    """
    Removes patterns like   from the     
    Args:
        text (str): The input string containing potential patterns to remove.
    
    Returns:
        str: The cleaned text without the specified patterns.
    """
    # Define the regular expression to match the pattern
    pattern = r"【\d+:\d+†source】"
    
    # Use re.sub to remove all occurrences of the pattern
    cleaned_text = re.sub(pattern, "", text)
    
    # Return the cleaned text
    return cleaned_text


def document_filler(doc_copy, prompt_name, assistant_response):
    #First we loop through all the paragraphs.
    for paragraph in doc_copy.paragraphs:

        #If the prompt_name correspond to the placeholder making up the paragraph
        #we move to the filling part
        if prompt_name in paragraph.text:
            
            #This is for formatting reasons to avoid alignment problems
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

            #Then, we move to the run objects inside the paragraph.
            #The reason is that in this way, when we replace the placeholder 
            #we will keep the placeholder's formatting
            for run in paragraph.runs:
                if prompt_name in run.text:
                    run.text = run.text.replace(prompt_name, assistant_response)

def adding_headers(document, title):

    section = document.sections[0]

    # Access the header of the section
    header = section.header

    paragraph = header.paragraphs[0]

    new_text = []

    left_paragraph= f"Project {title.capitalize()}"
    
    new_text.append(left_paragraph)

    current_date = datetime.now()

    right_paragraph = current_date.strftime("%B %Y").capitalize()
    #print(right_paragraph)

    new_text.append(right_paragraph)

    #paragraph.text = f"{left_paragraph}\t\t{right_paragraph}"
    
    l = ['Project', 'Date']

    for sub, new in zip(l, new_text):

        for run in paragraph.runs:
                    print(run.text)
                    if sub in run.text:
                        run.text = run.text.replace(sub, new)



