import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize index
index = pc.Index("mathindex")

# Initialize embeddings and OpenAI client
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def query_pinecone(query_text, index, embeddings, top_k=3):
    try:
        query_vector = embeddings.embed_query(query_text)
        query_response = index.query(
            vector=query_vector, 
            top_k=top_k, 
            include_metadata=True
        )
        
        results = []
        for item in query_response['matches']:
            result = {
                'id': item.get('id', 'No ID'),
                'score': item.get('score', 'No score'),
                'text': item.get('metadata', {}).get('text', 'No text available')
            }
            results.append(result)
            print(item.get('id', 'No ID'))
        
        return results
    
    except Exception as e:
        st.error(f"An error occurred while querying Pinecone: {e}")
        return []

def generate_response(query, context):
    try:
        prompt = f"""As a mathematical assistant, use the following context to answer the user's question. 
        If the context doesn't provide enough information, use your general knowledge about mathematics, 
        but prioritize the given context. If the user's question is about mathematics, make sure to respond in the language that the question was asked with.
        
        
        When responding:
        - Ensure that all LaTeX expressions are correctly formatted.
        - Correct any LaTeX errors and provide properly formatted LaTeX (e.g., 
        - incorrect : \[ \chi^2 = \frac{{(n-1)s^2}}{{\sigma_0^2}} \], 
        - correct :
        $$
        \[ \chi^2 = \frac{{(n-1)s^2}}{{\sigma_0^2}} \]
        $$).
        Note: Make sure to use proper LaTeX delimiters (e.g., $$ for display mode or $ for inline mode) and ensure that any LaTeX code is correctly rendered without errors.

        

        Context:
        {context}
        Ensure correct LaTeX formatting : $latex_expr$
        User's question: {query}

        Assistant's response:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful probability assistant.Ensure correct LaTeX formatting : $latex_expr$"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "I apologize, but I encountered an error while processing your question. Could you please try asking again?"

# def display_response(response):
#     # Pattern to match inline ($...$), display ($$...$$), and custom display ([...]) LaTeX
#     pattern = r'(\$\$.*?\$\$|\$.*?\$|\[.*?\])'

#     # Split the response into LaTeX and non-LaTeX parts
#     parts = re.split(pattern, response)
    
#     for part in parts:
#         part = part.strip()
#         if part.startswith('$$') and part.endswith('$$'):
#             # Display LaTeX (equations on separate lines)
#             st.latex(part[2:-2].replace('\\ ', ' ').replace('\n', ' '))  # Remove $$ delimiters
#         elif part.startswith('$') and part.endswith('$'):
#             # Inline LaTeX
#             st.markdown(f"${part[1:-1].replace('\\ ', ' ').replace('\n', ' ')}$")
#         elif part.startswith('[') and part.endswith(']'):
#             # Custom display LaTeX using []
#             st.latex(part[1:-1].replace('\\ ', ' ').replace('\n', ' '))
#         else:
#             # Regular text
#             clean_part = part.replace('\\ ', ' ').replace('\n', ' ')
#             st.markdown(clean_part)

# Streamlit UI
st.title("Proba Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # display_response(message["content"])
            st.write(message["content"])
        else:
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    results = query_pinecone(prompt, index, embeddings, top_k=3)
    
    if results:
        context = "\n".join([result['text'] for result in results])
        response = generate_response(prompt, context)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # display_response(response)
            st.write(response)
            print(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            st.markdown("I'm sorry, but I couldn't find any relevant information to answer your question. Could you please rephrase or ask a different question?")
        st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, but I couldn't find any relevant information to answer your question. Could you please rephrase or ask a different question?"})

# Add a sidebar with some information
st.sidebar.title("About")
st.sidebar.info("This is a Proba Chatbot powered by Pinecone and OpenAI provided by Esprit . Ask any Probability-related question!")
st.sidebar.warning("Note: This chatbot's knowledge is based on the information indexed in your Courses and in your TDs.")
