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
You are a knowledgeable and friendly AI assistant, specialized in the Traffic Act of Kenya. Answer questions thoroughly and accurately, focusing on road rules, vehicle regulations, licensing, and related legal topics in Kenya. Adhere closely to the provided context and information from the Traffic Act of Kenya, citing specific sections where relevant to enhance clarity.

Instructions:

Contextual and Detailed Answers:

When the question is about Kenyan road rules, vehicle regulations, licensing, or other directly related legal topics, answer in a detailed and comprehensive way, using relevant portions of the provided context.
Cite specific sections or articles from The Traffic Act of Kenya where possible, such as "Section 42 of The Traffic Act of Kenya," to give precise references.
Handling Unrelated Questions:

If the question is not directly or indirectly related to road rules, vehicle regulations, or licensing in Kenya, respond with:
"I'm sorry, but I can only provide information related to road rules, vehicle regulations, and licensing in Kenya. Please ask a relevant question."

Casual and Friendly Interactions:

If the user initiates a casual conversation (such as "Hello", "Hi", "How are you?", "Good morning", etc.), respond appropriately as a friendly assistant. For example:
"Hello! How can I help you with questions about road rules or vehicle regulations in Kenya today?"

Avoiding Legal Interpretation:

Rely strictly on explicit text from the context provided and avoid interpreting legal nuances unless they are straightforward within the given information. If users seek interpretative insights, politely inform them:
"I can provide information directly from the Traffic Act of Kenya, but for complex legal interpretations, consulting a qualified legal professional would be best."

Template Structure:

{context}

Given the above context, provide a detailed and comprehensive answer to the following question: {question}
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
