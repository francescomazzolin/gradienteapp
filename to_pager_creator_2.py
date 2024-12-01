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

import openai

import pandas as pd
import re
import time
import pickle
import importlib

import to_pager_functions_2 as fc

importlib.reload(fc)

"""
SCRIPTING STUFF
"""

temp_responses = []

answers_dict = {}


"""
=======================================================================================================================
Beginning of the script: Creating the connection with the OpenAI API 
=======================================================================================================================
"""

"""
Setting up the OpenAI connection by loading the API key and creating a client
"""

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check API key
if openai.api_key is None:
    print("Error: OPENAI_API_KEY not found. Make sure the .env file is loaded properly.")
    exit()

# Initialize the OpenAI client
client = openai.OpenAI()

"""
This retrieves the .docx template that will be filled with the responses to our prompts.

This file contains:

    1) Sections titles

    2) Placeholders: These will be replaced by their corresponding prompt answers
"""

template_path = 'to_pager_template.docx'
doc_copy = Document(template_path)


"""
=================================================================================================================
A.	BUSINESS OPPORTUNITY AND GROUP OVERVIEW
=================================================================================================================
"""

assistant_identifier = 'asst_ZwYHPxoqquAdDHmVyZrr8SgC'

prompt_list, additional_formatting_requirements, prompt_df = fc.prompts_retriever('prompt_db.xlsx', 
                                                                                  ['BO_Prompts', 'BO_Format_add'])

for prompt_name, prompt_message in prompt_list:

    prompt_message = fc.prompt_creator(prompt_df, prompt_name, 
                                       prompt_message, additional_formatting_requirements,
                                       answers_dict)

    assistant_response = fc.separate_thread_answers(client, prompt_message, 
                                                    assistant_identifier)

    if assistant_response:
        
        print(f"Assistant response for prompt '{prompt_name}': {assistant_response}")

        temp_responses.append(assistant_response)

        assistant_response = fc.remove_source_patterns(assistant_response)

        answers_dict[prompt_name] = assistant_response

        fc.document_filler(doc_copy, prompt_name, assistant_response)

# Save the modified document
#new_file_path = 'business_overview.docx'
#doc_copy.save(new_file_path)
#print(f"Modified document saved as {new_file_path}")

"""
=================================================================================================================
B.	REFERENCE MARKET
=================================================================================================================
"""


assistant_identifier = 'asst_vy2MqKVgrmjCecSTRgg0y6oO'


prompt_list, additional_formatting_requirements, prompt_df = fc.prompts_retriever('prompt_db.xlsx', 
                                                                                  ['RM_Prompts', 'RM_Format_add'])
for prompt_name, prompt_message in prompt_list:

    prompt_message = fc.prompt_creator(prompt_df, prompt_name, 
                                       prompt_message, additional_formatting_requirements,
                                       answers_dict)

    assistant_response = fc.separate_thread_answers(client, prompt_message, 
                                                    assistant_identifier)
    

    if assistant_response:
        print(f"Assistant response for prompt '{prompt_name}': {assistant_response}")

        temp_responses.append(assistant_response)

        assistant_response = fc.remove_source_patterns(assistant_response)

        answers_dict[prompt_name] = assistant_response

        fc.document_filler(doc_copy, prompt_name, assistant_response)


# Save the modified document
new_file_path = 'to_pager_official_3.docx'
doc_copy.save(new_file_path)
print(f"Modified document saved as {new_file_path}")
