import asyncio
import cohere
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate


CHROMA_PATH = 'chroma_r'

# Load API keys from Streamlit secrets
cohere_key = st.secrets.credentials.cohere_key
google_key = st.secrets.credentials.google_key

#Configure GenAI 
genai.configure(api_key = google_key)

# Initialize Cohere client
co = cohere.Client(cohere_key)

class CohereEmbeddings:
    def __init__(self, model='embed-english-v2.0'):
        self.model = model

    def embed_documents(self, texts):
        response = co.embed(texts=texts, model=self.model)
        return response.embeddings

    def embed_query(self, query):
        return self.embed_documents([query])[0]
    
llm = genai.GenerativeModel('gemini-pro')

# Initialize Chroma with custom embedding function
embedding_function = CohereEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# prompt = "Is overtaking illegal"

st.title('StreetSmartKE🚸')

def get_response(prompt):
    result = db._similarity_search_with_relevance_scores(prompt)
    # print(result)

    # Define prompt template
    PROMPT_TEMPLATE = """
    You are an expert AI assistant. Answer the following question in detail, providing as much context and explanation as possible based on the provided information, while clearly stating that it is from The Traffic Act of Kenya and citing sources:

    {context}

    ---

    Given the above context, provide a detailed and comprehensive answer to the following question: {question}

    If the question is unrelated to The Traffic Act of Kenya or if there is insufficient context to answer the question, please respond with: "I'm sorry, but I can only provide information related to The Traffic Act of Kenya. Please ask a relevant question."
    """


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt_template.format(context=context_text, question=prompt)
    

    response = llm.generate_content(formatted_prompt)
    print(response.text)
    st.info(response.text)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        get_response(text)