#%%
# PDF Chatbot Libraries
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import openai
from bs4 import BeautifulSoup


# Supporting Functions for Chatbot
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    return text

def get_html_text(html_docs):
    text = ""
    for html_file in html_docs:
        try:
            html_content = html_file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            text += soup.get_text(separator="\n")
        except Exception as e:
            st.error(f"Error processing {html_file.name}: {e}")
    return text

def get_text_from_files(files):
    text = ""
    pdf_files = [file for file in files if file.name.endswith('.pdf')]
    html_files = [file for file in files if file.name.endswith('.html')]

    if pdf_files:
        text += get_pdf_text(pdf_files)
    if html_files:
        text += get_html_text(html_files)

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0.1,
        openai_api_key=openai.api_key
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first!")
        return

    # Get the response from the conversation chain
    response = st.session_state.conversation({'question': user_question})
    answer = response['answer']  # Assuming response contains an 'answer' key

    # Display the response in Streamlit
    st.write(answer)
