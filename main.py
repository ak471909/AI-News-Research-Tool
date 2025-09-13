import os
import streamlit as st
import time
import requests
from urllib.parse import urlparse
import shutil
import tempfile

# Updated imports for newer versions
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import numpy as np

from dotenv import load_dotenv
load_dotenv()

st.title("AI News Research Tool")
st.sidebar.title("News article URLs")

# Initialize session state for URLs if not exists
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set your OPENAI_API_KEY in the .env file")
    st.stop()

def is_valid_url(url):
    """Check if URL is valid and accessible"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Quick check if URL is accessible
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, timeout=10, allow_redirects=True, headers=headers)
        return response.status_code == 200
    except:
        return False

def load_url_content(url):
    """Load content from a single URL with enhanced error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    try:
        # Try with session for better compatibility
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Basic check if we got blocked
        if len(response.text) < 100 or "blocked" in response.text.lower() or "captcha" in response.text.lower():
            raise requests.RequestException("Site appears to be blocking automated access")
        
        # Create a simple document object
        return Document(
            page_content=response.text,
            metadata={"source": url}
        )
    except Exception as e:
        st.warning(f"Failed to load {url}: {str(e)}")
        return None

# Initialize LLM with better error handling
try:
    llm = OpenAI(
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=500,
        request_timeout=60
    )
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {str(e)}")
    st.stop()

if process_url_clicked:
    # Filter out empty URLs and validate them
    valid_urls = [url.strip() for url in urls if url.strip()]
    
    if not valid_urls:
        st.error("Please enter at least one valid URL")
    else:
        try:
            main_placeholder.text("Validating URLs...")
            
            # Check URL validity first
            accessible_urls = []
            for url in valid_urls:
                if is_valid_url(url):
                    accessible_urls.append(url)
                else:
                    st.warning(f"URL not accessible or invalid: {url}")
            
            if not accessible_urls:
                st.error("No accessible URLs found. Please check your URLs.")
            else:
                # Load data with progress tracking
                main_placeholder.text("Loading data from URLs...")
                documents = []
                
                progress_bar = st.progress(0)
                for i, url in enumerate(accessible_urls):
                    main_placeholder.text(f"Loading {i+1}/{len(accessible_urls)}: {url[:50]}...")
                    doc = load_url_content(url)
                    if doc:
                        documents.append(doc)
                    progress_bar.progress((i + 1) / len(accessible_urls))
                
                progress_bar.empty()
                
                if not documents:
                    st.error("No content could be loaded from the provided URLs")
                else:
                    # Split documents
                    main_placeholder.text("Splitting text into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ". ", " ", ""],
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    
                    docs = text_splitter.split_documents(documents)
                    st.info(f"Created {len(docs)} text chunks from {len(documents)} documents")
                    
                    # Create embeddings and vector store in memory
                    main_placeholder.text("Creating embeddings...")
                    try:
                        embeddings = OpenAIEmbeddings(
                            openai_api_key=api_key,
                            request_timeout=60
                        )
                        
                        main_placeholder.text("Building vector store...")
                        # Store in session state instead of persisting to disk
                        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                        
                        st.session_state.processed_urls = accessible_urls
                        main_placeholder.text("Processing completed successfully!")
                        time.sleep(2)
                        main_placeholder.empty()
                        st.success("URLs processed successfully! You can now ask questions.")
                        
                    except Exception as e:
                        st.error(f"Error creating embeddings: {str(e)}")
                        st.info("Try installing FAISS with: pip install faiss-cpu")
                        main_placeholder.empty()
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            main_placeholder.empty()

# Display processed URLs
if st.session_state.processed_urls:
    with st.sidebar.expander("Processed URLs"):
        for url in st.session_state.processed_urls:
            st.write(f"✅ {url}")

# Query section
st.markdown("---")
query = st.text_input("Ask a question about the processed articles:")

if query:
    if st.session_state.vectorstore is not None:
        try:
            # Create the chain with timeout
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, 
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            
            with st.spinner("Searching for answer..."):
                result = chain.invoke({"question": query})
            
            st.header("Answer")
            answer = result.get("answer", "No answer found")
            st.write(answer)
            
            # Display sources
            sources = result.get("sources", "")
            if sources and sources.strip():
                st.subheader("Sources")
                sources_list = [s.strip() for s in sources.split("\n") if s.strip()]
                for source in sources_list:
                    st.write(f"🔗 {source}")
            else:
                st.info("No specific sources were identified for this answer.")
                
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            st.info("Try processing the URLs again or check your OpenAI API key.")
    else:
        st.info("Please process some URLs first before asking questions.")

# Add some usage instructions
with st.expander("How to use this tool"):
    st.write("""
    1. **Enter URLs**: Add up to 3 news article URLs in the sidebar
    2. **Process URLs**: Click 'Process URLs' to load and analyze the content
    3. **Ask Questions**: Once processing is complete, ask questions about the articles
    4. **Get Answers**: The AI will provide answers based on the article content with sources
    
    **Tips**:
    - Make sure URLs are accessible and contain readable content
    - Wait for processing to complete before asking questions
    - Be specific in your questions for better results
    - The vector store is stored in memory - it will be cleared when you refresh the page
    """)

# Add clear data button
if st.sidebar.button("Clear Processed Data"):
    st.session_state.vectorstore = None
    st.session_state.processed_urls = []
    st.sidebar.success("Data cleared!")