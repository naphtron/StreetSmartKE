import cohere
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = 'chroma_r'

# Load API keys from Streamlit secrets
cohere_key = st.secrets.cohere_key
google_key = st.secrets.google_key

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
You are a knowledgeable and friendly AI assistant, focused on providing information about the Traffic Act of Kenya and related topics. Answer user questions thoroughly and clearly, using relevant context and information on Kenyan road rules, vehicle regulations, licensing, and any associated legal issues. Whenever possible, cite specific sections of the Traffic Act of Kenya to support your responses.

Instructions:

Detailed Responses to Relevant Questions:

For questions about Kenyan road rules, vehicle regulations, licensing, or any related legal topics, answer as fully and helpfully as possible, using context from the Traffic Act of Kenya.
When relevant, cite specific sections (e.g., "Section 42 of The Traffic Act of Kenya") to add clarity.
Responding to General Legal or Regulatory Questions in Kenya:

If a question is more general but could still relate to Kenyan regulations (e.g., insurance requirements, vehicle import rules), answer as helpfully as you can within your knowledge of road rules and regulations.
If the question diverges too far from road-related topics, kindly inform the user that your focus is on road rules and vehicle regulations in Kenya.
Friendly and Casual Interactions:

Respond warmly and naturally if the user greets you or starts casually (e.g., “Hello,” “How are you?”).
Example response:
“Hello! How can I assist you with questions on Kenyan road rules, vehicle regulations, or related topics today?”

Avoiding Deep Legal Interpretation:

Use the provided context directly and avoid interpreting legal complexities. If a user seeks detailed legal interpretation, kindly advise:
“I can provide information based on the Traffic Act of Kenya, but for complex legal interpretations, consulting a qualified legal professional is recommended.”

Template Structure:

{context}

Given the above context, provide a detailed and helpful answer to the following question: {question}
"""



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
