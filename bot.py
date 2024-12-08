import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load the PDF and extract text
pdf_path = "en3_canadian_charter_qc.pdf"  # Path to the Canadian Charter PDF
charter_text = extract_text_from_pdf(pdf_path)

# Initialize the Groq client
client = Groq()

# Streamlit app setup
st.set_page_config(page_title="Canadian Charter RAG Assistant", page_icon="📜", layout="wide")
st.title("📜 Canadian Charter of Rights and Freedoms Assistant")
st.markdown("""
### Ask questions about the Canadian Charter of Rights and Freedoms

This tool provides answers based on the Canadian Charter. Simply type your question below, and the assistant will retrieve relevant information and provide an accurate response.
""")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Your Question:", placeholder="Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Define the query for the RAG model
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "assistant", "content": (
                "You are an intelligent retrieval-augmented generation (RAG) system designed to assist users in "
                "understanding and exploring the Canadian Charter of Rights and Freedoms. Your task is to provide "
                "clear, concise, and accurate answers to user queries by retrieving relevant information from the Charter "
                "and summarizing or elaborating as needed. Ensure your responses are contextually relevant, factual, and "
                "grounded in the Charter's content. If the query cannot be answered directly from the Charter, politely "
                "indicate this and avoid speculation."
            )},
            {"role": "assistant", "content": charter_text},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        stream=True,
        stop=None,
    )

    # Stream the model's response
    response = ""
    with st.spinner("Generating response..."):
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                response += content
        st.text_area("Assistant Response:", response, height=200, key=f"response_{len(st.session_state.messages)}")

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.write(f"**You:** {msg['content']}", key=f"user_{i}")
    else:
        st.write(f"**Assistant:** {msg['content']}", key=f"assistant_{i}")
