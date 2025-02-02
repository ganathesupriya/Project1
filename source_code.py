import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure page
st.set_page_config(
    page_title="AI-Powered Document Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

def check_api_keys():
    """Check if API keys are set in session state."""
    return (
        'groq_api_key' in st.session_state 
        and 'google_api_key' in st.session_state 
        and st.session_state.groq_api_key 
        and st.session_state.google_api_key
    )

def set_api_keys():
    """Set API keys in session state."""
    st.sidebar.header("🔑 API Key Configuration")

    # Hard-code the API keys here
    groq_key = "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"
    google_key = "AIzaSyDtogdxM4-e_rgdU-PdH61jPjVwiZ42xIs"
    
    # Set them directly in the session state
    st.session_state.groq_api_key = groq_key
    st.session_state.google_api_key = google_key
    st.session_state.api_keys_set = True
    
    # Optionally set the environment variables as well
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["GROQ_API_KEY"] = groq_key
    
    # Display success message
    st.sidebar.success("✅ API keys have been set successfully!")

def initialize_vector_store():
    """Initialize the vector store with document embeddings."""
    with st.spinner("📚 Loading and processing documents..."):
        try:
            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=st.session_state.google_api_key)
            
            # Validate folder path
            folder_path = "documents"
            if not os.path.exists(folder_path):
                st.error(f"❌ Folder not found: {folder_path}")
                return
            
            # Collect all PDF files in the folder
            pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
            if not pdf_files:
                st.error("❌ No PDF files found in the specified folder.")
                return

            # Load and process all PDFs
            all_docs = []
            for pdf_file in pdf_files:
                loader = PyMuPDFLoader(pdf_file)
                docs = loader.load()
                all_docs.extend(docs)  # Combine all documents

            st.write(f"🔍 Total documents loaded: {len(all_docs)}")
            if not all_docs:
                st.error("❌ No content extracted from the uploaded PDFs.")
                return

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(all_docs)

            st.write(f"🔍 Total split documents: {len(split_docs)}")
            if not split_docs:
                st.error("❌ No content to process after splitting documents.")
                return
            
            # Create vector store
            vector_store = FAISS.from_documents(split_docs, embeddings)
            
            # Store in session state
            st.session_state.vectors = vector_store
            st.session_state.initialized = True
            st.success("✅ System initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing vector store: {str(e)}")

def main():
    # Page title and description
    st.title("🤖 AI-Powered Document Q&A Assistant")
    st.markdown("""
    Upload your documents and ask questions to get precise, context-aware answers.
    Start by configuring your API keys in the sidebar.
    """)

    # Set up API keys (hardcoded now)
    set_api_keys()
    
    if not st.session_state.initialized and st.session_state.api_keys_set:
        if st.button("🚀 Initialize System"):
            if check_api_keys():
                initialize_vector_store()
            else:
                st.warning("⚠️ Please set your API keys first.")

    prompt_input = st.text_input("🖍 Ask a question about your documents:")
    
    if prompt_input and st.session_state.initialized:
        with st.spinner("🤖 Generating your answer..."):
            try:
                # Create LLM instance
                llm = ChatGroq(
                    groq_api_key=st.session_state.groq_api_key,
                    model_name="mixtral-8x7b-32768"
                )
                
                # Create chains
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the provided context.
                Context: {context}
                Question: {input}
                Provide a detailed answer.
                """)
                
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Get response
                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': prompt_input})
                end_time = time.process_time()
                
                # Display response
                st.markdown("### 📋 Answer")
                st.write(response['answer'])
                st.info(f"⚡ Response time: {end_time - start_time:.2f} seconds")
                
                # Show relevant documents
                with st.expander("📑 Relevant Document Sections"):
                    for i, doc in enumerate(response.get("context", []), 1):
                        st.markdown(f"**Segment {i}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")

    elif prompt_input and not st.session_state.initialized:
        st.warning("⚠️ Please initialize the system first.")

if __name__ == "__main__":
    main()
