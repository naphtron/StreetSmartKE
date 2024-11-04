import cohere
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# Streamlit UI
st.title('StreetSmartKE🚸')
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about traffic laws in Kenya:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform similarity search
    results = db.similarity_search_with_relevance_scores(prompt, k=3)

    # Define prompt template
    PROMPT_TEMPLATE = """
You are an expert AI assistant. Answer the following question in detail, providing as much context and explanation as possible based on the provided information, while clearly stating that it is from The Traffic Act of Kenya and citing sources:

{context}

---

Given the above context, provide a detailed and comprehensive answer to the following question: {question}

However, if the question is a greeting or casual conversation starter (such as "Hello", "Hi", "Hey", "How are you?", "Good morning", and all the rest), please respond appropriately as a friendly assistant would.
"""

# caveat = If the question is unrelated to road, please respond with: "I'm sorry, but I can only provide information related to road rules and regulations in Kenya. Please ask a relevant question." 


    if len(results) == 0:
        with st.chat_message("assistant"):
            st.info("Unable to find matching results.")
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        formatted_prompt = prompt_template.format(context=context_text, question=prompt)
        
        # Generate response from the model
        
        response = llm.generate_content(formatted_prompt)

        with st.chat_message("assistant"):
            st.markdown(response.text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.text})
