import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from populate_database import (
    clear_database,
    add_to_chroma,
    calculate_chunk_ids,
    split_documents,
)
from langchain.schema.document import Document
from query_data import query_rag

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Sidebar contents
with st.sidebar:
    st.title("LLM Chat App")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Ollama](https://ollama.ai/)
    """
    )
    st.write("Made with streamlit")


def main():
    st.header("Chat with PDF")

    reset = st.sidebar.button("Reset Database")
    if reset:
        clear_database()
        st.write("Database has been reset.")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        with st.spinner("Processing PDF..."):
            try:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500, chunk_overlap=200, length_function=len
                )
                chunks = [
                    Document(
                        page_content=chunk, metadata={"source": pdf.name, "page": i}
                    )
                    for i, chunk in enumerate(text_splitter.split_text(text))
                ]

                add_to_chroma(chunks)
                st.success("PDF uploaded and processed successfully.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    model_name = st.selectbox(
        "Select Model", ["mistral", "llama2"], key="model_selectbox"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("Ask questions about your PDF file:")
    send_button = st.button("Send")

    if send_button and user_input:
        with st.spinner("Generating response..."):
            try:
                # Prepare history for the prompt
                history = "\n".join(
                    [
                        f"User: {q}\nChatbot: {r}"
                        for q, r, _ in st.session_state.conversation
                    ]
                )
                response = query_rag(user_input, model_name, history)
                st.session_state.conversation.append((user_input, response, model_name))
                user_input = ""
            except Exception as e:
                st.error(f"Error generating response: {e}")

    for q, r, model in st.session_state.conversation:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")
        st.write(f"**Model Used:** {model}")

    if st.session_state.conversation:
        st.write("**Bot:** Do you have more queries?")


if __name__ == "__main__":
    main()
