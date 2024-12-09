import openai
from openai import OpenAI
import streamlit as st
import time
import io
import os
from langchain.prompts import PromptTemplate

client = OpenAI()

assistant_id="asst_zPBNRyNnFpZKEEo8amPapEIB"    
vector_store_id = "vs_GzXAPsk7uFfiwCC4jVTjCENQ"

if "id_file_list" not in st.session_state:
    st.session_state.ids_file_list = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "vector_store_id" not in st.session_state:
    st.session_state["vector_store_id"] = vector_store_id

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

general_prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template=("You are an assistant able to generate plot based on Matplotlib, Seaborn and Plotly"
              "All the plot that you create must use these HTML colors : #003967, #1A4C77, #346688, #4F8098, #6999A8, #83B3B8"
              "All the years in the axis (if there are) must be integers, for example: 2020, 2021, 1980 ecc..., you can't write 2020.5,2021.0, this is not allowed"
              "Be extremely precise in creating the plot"
              "Remember the output must be an image not a text an image"
              "Use all these information to correctly plotting what is asked here:"
              "{user_input}"),
)

st.title("AI Plot Creator")
st.subheader("Based on the input file you can ask to Generate plots")


def upload_file_openai(filepath):
    with open(filepath,"rb") as file:
        response = client.files.create(file = file.read(), purpose="assistants")
        return response.id


st.sidebar.subheader("Vector Store Management")


if st.sidebar.button("Create Vector Store"):
    if st.session_state["vector_store_id"]:
        st.sidebar.info(f"Vector store already exists: {st.session_state['vector_store_id']}")
        st.write(f"Using existing Vector Store ID: {st.session_state['vector_store_id']}")
    else:
        vector_store = client.beta.vector_stores.create(name="Graph Vector")
        st.session_state["vector_store_id"] = vector_store.id
        st.sidebar.success(f"Vector store created: {vector_store.id}")
        st.write(f"Vector store created with ID: {vector_store.id}")




file_upload = st.sidebar.file_uploader("Upload a PDF or Text File", type=["pdf", "txt","csv","xlxs"])

if file_upload:
    local_file_path = file_upload.name
    with open(local_file_path, "wb") as f:
        f.write(file_upload.getbuffer())
    st.session_state.uploaded_file_name = local_file_path
    st.sidebar.success(f"File uploaded and saved locally: {local_file_path}")
    

if st.session_state["vector_store_id"] and st.session_state["uploaded_file_name"]:
    if st.sidebar.button("Upload to Vector Store"):
        try:
            with open(st.session_state["uploaded_file_name"], "rb") as file_stream:
                file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=st.session_state["vector_store_id"],
                    files=[file_stream]
                )

            st.sidebar.success("File successfully added to vector store.")
        except Exception as e:
            st.sidebar.error(f"Error uploading file: {e}")

#Button to initiate chat session

if st.sidebar.button("start Chatting"):
        st.session_state.start_chat = True
        
        #Create a new thread
        st.write("chat is starting")
        chat_thread = client.beta.threads.create()
        st.session_state.thread_id = chat_thread.id


# Function to extract file_id
if st.session_state.start_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if prompt:= st.chat_input("Ask a Question"):

        structured_prompt = general_prompt_template.format(
            user_input = prompt
        )        

        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role = "user",
            content=structured_prompt
        )

        run = client.beta.threads.runs.create(
            
            thread_id=st.session_state.thread_id,
            assistant_id=assistant_id
        )

        with st.spinner("Generatin Response..."):
            while run.status != "completed":
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                   thread_id=st.session_state.thread_id,
                   run_id= run.id
                )
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id

            )

            counter = 0
            last_image_path = None  # Track the path of the last image created

            for message in messages.data:
                if message.role == 'assistant':
                    counter += 1
                    counter_block = 0
                    for content_block in message.content:
                        if content_block.type == 'image_file':
                            # Retrieve the image file ID
                            image_id = content_block.image_file.file_id
                            
                            # Get the image data from the API
                            image_data = client.files.content(image_id)
                            image_data_bytes = image_data.read()

                            # Ensure a unique output directory
                            output_dir = "test"
                            os.makedirs(output_dir, exist_ok=True)

                            # Define a unique path to save the image
                            image_path = os.path.join(output_dir, f"plot_{counter}_{counter_block}.png")
                            counter_block += 1
                            if counter == 1:
                                last_image_path = image_path

                            # Save the image locally
                            with open(image_path, "wb") as f:
                                f.write(image_data_bytes)

                            # Update the last_image_path to track the most recent image
                            

            # Display only the last image created
            if last_image_path:
                st.image(last_image_path)
            else:
                st.warning("No images found in the assistant's response.")