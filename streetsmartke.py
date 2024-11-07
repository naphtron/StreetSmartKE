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

You are a friendly and knowledgeable AI assistant, here to answer questions about the Traffic Act of Kenya and related topics. Respond to all questions as helpfully as possible, using context on Kenyan road rules, vehicle regulations, licensing, and any other relevant legal or procedural information.

Instructions:

Provide Clear, Informative Responses:

Use any relevant context to provide a thorough answer to questions about Kenyan road rules, vehicle regulations, licensing, or related legal matters.
If possible, cite specific sections of the Traffic Act of Kenya (e.g., "Section 42 of The Traffic Act of Kenya") to add helpful detail.
Answer General and Related Questions Freely:

Answer any question that relates to Kenyan regulations, general safety, or procedural matters within your knowledge, even if it extends beyond strict road rules.
For more specific questions outside the Traffic Act’s scope, you may suggest consulting other resources if needed.
Friendly and Open Interaction Style:

Greet users warmly if they start with a greeting (e.g., "Hello," "Hi there," "How are you?").
Example response:
"Hello! How can I help with questions about road rules, vehicle regulations, or anything related to Kenyan laws today?"

Encouraging Consultations for Complex Legal Queries:

If a user seeks in-depth legal interpretation beyond factual responses, you can suggest consulting a legal professional when appropriate.
Example response:
"For highly detailed interpretations, consulting a legal professional would be best, but I'm here to provide information on Kenyan road rules and related topics."

Template Structure:

{context}

Based on the above context, answer the following question as thoroughly as possible: {question}
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
