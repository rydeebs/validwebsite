import streamlit as st

# Initialize session state variables at the very top of the script
if 'last_search_query' not in st.session_state:
    st.session_state['last_search_query'] = ""
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
if 'current_chunk' not in st.session_state:
    st.session_state['current_chunk'] = 0
if 'total_chunks' not in st.session_state:
    st.session_state['total_chunks'] = 0
if 'processing' not in st.session_state:
    st.session_state['processing'] = False
if 'stop_requested' not in st.session_state:
    st.session_state['stop_requested'] = False
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0

# Now import all other libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time
from googlesearch import search
import concurrent.futures
import io
import os
import socket
import random
import logging
import json
import hashlib
import base64
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to create a unique session ID
def get_session_id():
    if 'session_id' not in st.session_state:
        # Generate a random session ID
        st.session_state['session_id'] = base64.b64encode(os.urandom(16)).decode('utf-8')
    return st.session_state['session_id']

# Function to get cache directory
def get_cache_dir():
    cache_dir = Path('./.streamlit/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

# Function to generate a cache key for a dataframe
def get_df_hash(df):
    # Use a sample of the dataframe to create a hash
    sample_size = min(100, len(df))
    sample = df.sample(sample_size) if len(df) > sample_size else df
    df_json = sample.to_json(orient='records')
    return hashlib.md5(df_json.encode()).hexdigest()

# Function to save progress
def save_progress(df, results, current_index, session_id, df_hash):
    cache_dir = get_cache_dir()
    progress_file = cache_dir / f"progress_{session_id}_{df_hash}.json"
    
    progress_data = {
        'current_index': current_index,
        'results': results,
        'timestamp': time.time()
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)
    
    logger.info(f"Progress saved at index {current_index}")

# Function to load progress
def load_progress(session_id, df_hash):
    cache_dir = get_cache_dir()
    progress_file = cache_dir / f"progress_{session_id}_{df_hash}.json"
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            logger.info(f"Loaded progress from index {progress_data['current_index']}")
            return progress_data['current_index'], progress_data['results']
        except Exception as e:
            logger.error(f"Error loading progress: {str(e)}")
    
    return 0, []

# Function to clear progress
def clear_progress(session_id, df_hash):
    cache_dir = get_cache_dir()
    progress_file = cache_dir / f"progress_{session_id}_{df_hash}.json"
    
    if progress_file.exists():
        try:
            os.remove(progress_file)
            logger.info("Progress file cleared")
        except Exception as e:
            logger.error(f"Error clearing progress: {str(e)}")

# Function to chunk a dataframe
def chunk_dataframe(df, chunk_size=1000):
    num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division
    return [df[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

# Function to check if a URL is valid
def check_url(url):
    if not url:
        return False, "Empty URL"
    
    # Add http:// if missing
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
        }
        
        # First check if domain resolves (DNS check)
        try:
            domain = urlparse(url).netloc
            socket.gethostbyname(domain)
        except socket.gaierror:
            return False, f"DNS resolution failed: Cannot resolve hostname '{domain}'"
        
        # Now attempt to connect with improved timeout and retry settings
        response = requests.get(
            url, 
            timeout=(5, 20),  # (connect timeout, read timeout)
            headers=headers, 
            allow_redirects=True,
            verify=True       # SSL verification
        )
        
        # Check if response is a success (200 OK)
        if response.status_code == 200:
            # Check if the page is not a common error page
            content = response.text.lower()
            error_indicators = [
                '404 not found', 
                'page not found', 
                'site not found', 
                'access denied',
                'forbidden',
                'error 404',
                'server error',
                'service unavailable'
            ]
            if any(term in content for term in error_indicators):
                return False, f"Page content indicates error: Status code {response.status_code}"
            
            # Check for very small responses which might be error pages
            if len(content) < 500:
                if not any(term in content for term in ['redirect', 'loading']):
                    return False, "Response too small, likely an error page"
                
            return True, response.url
        else:
            return False, f"HTTP Status code: {response.status_code}"
    
    except requests.exceptions.ConnectTimeout:
        return False, "Connection timed out while attempting to connect to the server"
    except requests.exceptions.ReadTimeout:
        return False, "Server took too long to respond (read timeout)"
    except requests.exceptions.SSLError:
        return False, "SSL certificate verification failed"
    except requests.exceptions.ConnectionError as e:
        if "RemoteDisconnected" in str(e):
            return False, "Remote server closed connection unexpectedly"
        elif "NameResolutionError" in str(e):
            return False, "Domain name resolution failed (DNS error)"
        else:
            return False, f"Connection error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

# Function to get company name from URL
def extract_company_name(url):
    """Extract the likely company name from a URL"""
    # Remove http://, https://, www.
    domain = urlparse(url).netloc if urlparse(url).netloc else url
    domain = domain.replace('www.', '')
    
    # Remove common TLDs and country codes
    domain = re.sub(r'\.(com|net|org|co|io|gov|edu|uk|us|ca|au|de|fr|jp|cn|in|br).*', '', domain)
    
    # Replace dashes and underscores with spaces
    domain = re.sub(r'[-_]', ' ', domain)
    
    # Capitalize each word
    words = domain.split()
    company_name = ' '.join(word.capitalize() for word in words)
    
    return company_name

def get_search_query(url, country=None):
    # Remove http://, https://, www.
    domain = urlparse(url).netloc if urlparse(url).netloc else url
    domain = domain.replace('www.', '')
    
    # Remove common TLDs and country codes
    domain = re.sub(r'\.(com|net|org|co|io|gov|edu|uk|us|ca|au|de|fr|jp|cn|in|br)', '', domain)
    
    # Replace symbols with spaces
    domain = re.sub(r'[-_]', ' ', domain)
    
    # Clean up the domain by removing common terms that aren't part of company name
    terms_to_remove = ['shop', 'store', 'online', 'official', 'site', 'website']
    domain_words = domain.split()
    domain_cleaned = ' '.join(word for word in domain_words if word.lower() not in terms_to_remove)
    
    # Build the search query
    search_query = domain_cleaned + " factory manufacturing supplier official website"
    
    # Add country to the search query if provided
    if country and isinstance(country, str) and len(country.strip()) > 0:
        # Clean up country name
        country = country.strip()
        # Add country to search query
        search_query += f" {country}"
    
    return search_query

# Function to find alternative URL via Google search
def find_alternative_url(url, country=None):
    company_name = extract_company_name(url)
    search_query = get_search_query(url, country)
    
    # Store for debugging purposes
    if 'last_search_query' in st.session_state:
        st.session_state['last_search_query'] = search_query
    
    # Try different search strategies
    search_attempts = [
        # First attempt: Full search query with company and industry terms
        {"query": search_query, "num": 5, "stop": 5},
        
        # Second attempt: Just company name + country + "official website"
        {"query": f"{company_name} {country if country else ''} official website", "num": 5, "stop": 5},
        
        # Third attempt: Just company name + "manufacturing" or "factory"
        {"query": f"{company_name} manufacturing factory", "num": 5, "stop": 5}
    ]
    
    for attempt in search_attempts:
        try:
            for result in search(attempt["query"], num=attempt["num"], stop=attempt["stop"], pause=2):
                try:
                    # Skip certain domains that are likely not company websites
                    if any(domain in result.lower() for domain in [
                        'facebook.com', 'linkedin.com', 'instagram.com', 'twitter.com', 
                        'youtube.com', 'pinterest.com', 'wikipedia.org', 'yelp.com',
                        'google.com', 'amazon.com', 'ebay.com', 'alibaba.com'
                    ]):
                        continue
                    
                    # Verify if the result is a valid website
                    is_valid, result_url = check_url(result)
                    if is_valid:
                        return result_url
                except Exception as e:
                    # If a specific search result fails, continue to the next one
                    continue
        except Exception as e:
            # If this search strategy fails, try the next one
            continue
    
    # If all strategies fail
    return "No valid alternative found"
  # Function to process a single row
def process_row(row_data, url_column, country_column=None, 
                connection_timeout=5, read_timeout=15, max_search_results=5,
                use_multiple_strategies=True, randomize_delay=True,
                progress_bar=None):
    try:
        url = row_data[url_column]
        result = {}
        
        # Copy all original data
        for col in row_data.index:
            result[col] = row_data[col]
        
        # Get country if country column is provided
        country = None
        if country_column and country_column in row_data:
            country = row_data[country_column]
        
        # Log the URL being processed
        logger.info(f"Processing URL: {url}")
        
        # Add random delay if enabled
        if randomize_delay:
            delay = random.uniform(0.5, 2.0)
            time.sleep(delay)
        
        # Check if URL is valid
        is_valid, message = check_url(url)
        result['Original URL'] = url
        result['Is Valid'] = is_valid
        result['Status Message'] = message
        
        # Store search parameters
        if country:
            result['Country Used'] = country
        
        # If not valid, find alternative
        if not is_valid:
            logger.info(f"URL invalid: {url}. Searching for alternative...")
            alternative_url = find_alternative_url(url, country)
            result['Alternative URL'] = alternative_url
            result['Final URL'] = alternative_url if alternative_url and not alternative_url.startswith("Search error") and not alternative_url == "No valid alternative found" else url
            
            # Log the result
            if alternative_url and not alternative_url.startswith("Search error") and not alternative_url == "No valid alternative found":
                logger.info(f"Found alternative URL: {alternative_url}")
            else:
                logger.info(f"No valid alternative found for: {url}")
        else:
            result['Alternative URL'] = ""
            result['Final URL'] = url
            logger.info(f"URL is valid: {url}")
        
        # Update progress bar if provided
        if progress_bar is not None:
            progress_bar.progress(1)
        
        return result
    
    except Exception as e:
        # Handle any unexpected errors during processing
        logger.error(f"Error processing row with URL {url if 'url' in locals() else 'unknown'}: {str(e)}")
        
        # Create a minimal result with error information
        error_result = {}
        for col in row_data.index:
            error_result[col] = row_data[col]
        
        error_result['Original URL'] = url if 'url' in locals() else row_data.get(url_column, "Unknown")
        error_result['Is Valid'] = False
        error_result['Status Message'] = f"Processing error: {str(e)}"
        error_result['Alternative URL'] = ""
        error_result['Final URL'] = url if 'url' in locals() else ""
        
        # Update progress bar
        if progress_bar is not None:
            progress_bar.progress(1)
        
        return error_result

def main():
    # Initialize session state for debugging
    if 'last_search_query' not in st.session_state:
        st.session_state['last_search_query'] = ""
    if 'processing_complete' not in st.session_state:
        st.session_state['processing_complete'] = False
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = None
    if 'current_chunk' not in st.session_state:
        st.session_state['current_chunk'] = 0
    if 'total_chunks' not in st.session_state:
        st.session_state['total_chunks'] = 0
    
    # Get session ID for caching
    session_id = get_session_id()
    
    st.title("Website URL Verifier")
    st.write("Upload a spreadsheet with website URLs to verify and find alternatives for invalid ones.")
    
    # File upload section
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Load preview of data first (just a few rows)
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file, nrows=5)
                # Reset file pointer for later full load
                uploaded_file.seek(0)
            else:
                preview_df = pd.read_excel(uploaded_file, nrows=5)
                uploaded_file.seek(0)
            
            st.write("File preview loaded successfully!")
            
            # Display preview
            st.subheader("Preview of the data")
            st.dataframe(preview_df.head())
            
            # Configuration section
            st.subheader("Configuration")
            
            # Select URL column
            url_column = st.selectbox(
                "Select the column containing website URLs",
                options=preview_df.columns.tolist()
            )
            
            # Select Country column (optional)
            col1, col2 = st.columns(2)
            with col1:
                use_country = st.checkbox("Use country information to improve search", value=True)
            
            country_column = None
            if use_country:
                with col2:
                    country_options = ["None"] + preview_df.columns.tolist()
                    country_column = st.selectbox(
                        "Select the column containing country information",
                        options=country_options
                    )
                    if country_column == "None":
                        country_column = None
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("Batch size (higher may be faster but could hit rate limits)", 
                                      min_value=1, max_value=20, value=3)
            with col2:
                max_workers = st.slider("Number of parallel workers", 
                                       min_value=1, max_value=5, value=2)
            
            # Large file handling options
            with st.expander("Large File Handling Options"):
                st.markdown("### Chunking Settings")
                chunk_size = st.slider("Chunk size (rows per processing segment)", 
                                      min_value=100, max_value=10000, value=1000)
                st.info("For very large files, processing is done in chunks to prevent timeouts. Each chunk is processed completely before moving to the next.")
                
                auto_save_frequency = st.slider("Auto-save frequency (rows)", 
                                      min_value=10, max_value=500, value=50)
                
                resume_processing = st.checkbox("Resume from last saved point if available", value=True)
                st.warning("If processing is interrupted, you can resume from the last saved point. Uncheck this to start fresh.")
            
            # Advanced options
            with st.expander("Advanced Connection Options"):
                st.markdown("### Connection Settings")
                connection_timeout = st.slider("Connection timeout (seconds)", 
                                      min_value=3, max_value=30, value=5)
                read_timeout = st.slider("Read timeout (seconds)", 
                                      min_value=5, max_value=60, value=15)
                
                st.markdown("### Search Enhancement")
                include_industry = st.checkbox("Include industry-specific terms in search", value=True)
                if include_industry:
                    industry_terms = st.text_input(
                        "Industry-specific search terms (comma separated)",
                        value="factory,manufacturing,supplier,producer"
                    )
                    
                st.markdown("### Google Search Settings")
                max_search_results = st.slider("Maximum search results to check per URL", 
                                      min_value=3, max_value=10, value=5)
                use_multiple_search_strategies = st.checkbox("Use multiple search strategies for better results", value=True)
                
                st.markdown("### Rate Limiting Protection")
                sleep_time = st.slider("Sleep time between batches (seconds)", 
                                       min_value=1, max_value=15, value=5)
                randomize_delay = st.checkbox("Add random delay between requests (recommended)", value=True)
                            
            # Process button logic
            start_button = st.button("Start/Resume Verification")
            stop_button = st.button("Stop Processing (Complete Current Batch)")
            
            # Add to session state to track if stop was requested
            if stop_button:
                st.session_state['stop_requested'] = True
                st.warning("Stop requested. Processing will halt after the current batch completes.")
            
            if start_button or ('processing' in st.session_state and st.session_state['processing']):
                # Mark as processing
                st.session_state['processing'] = True
                
                if start_button:
                    st.session_state['stop_requested'] = False
                
                # Show processing UI
                st.subheader("Processing")
                
                # Initialize progress containers
                overall_progress_container = st.empty()
                chunk_progress_container = st.empty()
                batch_progress_container = st.empty()
                status_text = st.empty()
                debug_log = st.empty()
                
                # Setup a handler to show logs in the Streamlit app
                log_output = io.StringIO()
                log_handler = logging.StreamHandler(log_output)
                log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(log_handler)
                
                # Now load the full file if we're starting or resuming
                if start_button or st.session_state.get('current_chunk', 0) == 0:
                    # Load the complete data
                    try:
                        with st.spinner("Loading complete dataset..."):
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file, low_memory=False)
                            else:
                                df = pd.read_excel(uploaded_file)
                            
                            st.success(f"Complete file loaded: {len(df)} rows and {len(df.columns)} columns")
                            logger.info(f"Loaded complete file with {len(df)} rows")
                            
                            # Create chunks for processing
                            df_chunks = chunk_dataframe(df, chunk_size)
                            st.session_state['total_chunks'] = len(df_chunks)
                            
                            # Calculate hash for this dataframe (for caching)
                            df_hash = get_df_hash(df)
                            st.session_state['df_hash'] = df_hash
                            
                            # Check if we should resume from a previous run
                            if resume_processing:
                                current_index, saved_results = load_progress(session_id, df_hash)
                                current_chunk = current_index // chunk_size
                                
                                if current_chunk > 0:
                                    st.session_state['current_chunk'] = current_chunk
                                    st.session_state['results'] = saved_results
                                    st.info(f"Resuming from chunk {current_chunk+1}/{len(df_chunks)} ({len(saved_results)} URLs already processed)")
                                else:
                                    st.session_state['current_chunk'] = 0
                                    st.session_state['results'] = []
                            else:
                                # Clear any existing progress
                                clear_progress(session_id, df_hash)
                                st.session_state['current_chunk'] = 0
                                st.session_state['results'] = []
                    
                    except Exception as e:
                        st.error(f"Error loading complete file: {str(e)}")
                        logger.error(f"File loading error: {str(e)}")
                        st.session_state['processing'] = False
                        return
                
                # Only proceed if we loaded data successfully
                if 'df_hash' in st.session_state:
                    # Process in chunks
                    try:
                        # Get current state
                        current_chunk = st.session_state['current_chunk']
                        total_chunks = st.session_state['total_chunks']
                        results = st.session_state.get('results', [])
                        
                        # Show overall progress
                        overall_progress = current_chunk / total_chunks
                        overall_progress_container.progress(overall_progress)
                        
                        # If we have completed all chunks, skip to results
                        if current_chunk >= total_chunks:
                            st.session_state['processing'] = False
                            st.session_state['processing_complete'] = True
                            
                            # Convert results to DataFrame
                            results_df = pd.DataFrame(results)
                            st.session_state['results_df'] = results_df
                            
                            status_text.success(f"All chunks completed ({len(results)} URLs processed)")
                            overall_progress_container.progress(1.0)
                            
                            # Jump to displaying results
                            st.experimental_rerun()
                            return
                        
                        # Otherwise process the current chunk
                        status_text.info(f"Processing chunk {current_chunk+1} of {total_chunks}")
                        
                        # Get the chunk to process
                        chunk_df = chunk_dataframe(df, chunk_size)[current_chunk]
                        total_rows_in_chunk = len(chunk_df)
                        
                        # Initialize chunk progress
                        chunk_results = []
                        chunk_progress = 0
                        chunk_progress_container.progress(chunk_progress)
                        
                        # Process in batches with parallel execution
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            for i in range(0, total_rows_in_chunk, batch_size):
                                # Check if stop was requested
                                if st.session_state.get('stop_requested', False):
                                    status_text.warning("Processing stopped as requested. You can resume later.")
                                    
                                    # Save progress
                                    current_overall_index = current_chunk * chunk_size + i
                                    save_progress(df, results + chunk_results, current_overall_index, session_id, st.session_state['df_hash'])
                                    
                                    st.session_state['processing'] = False
                                    return
                                
                                batch_df = chunk_df.iloc[i:min(i+batch_size, total_rows_in_chunk)]
                                
                                # Update status
                                current_batch = f"{i+1} to {min(i+batch_size, total_rows_in_chunk)}"
                                progress_text = f"Chunk {current_chunk+1}/{total_chunks}, Rows"
                                status_text.text(f"{progress_text} {current_batch} of {total_rows_in_chunk}")
                                logger.info(f"Processing batch: chunk {current_chunk+1}, rows {current_batch}")
                                
                                # Reset batch progress bar
                                batch_progress_container.progress(0)
                                
                                # Submit batch for processing with additional parameters
                                future_to_row = {
                                    executor.submit(
                                        process_row, 
                                        row, 
                                        url_column, 
                                        country_column,
                                        connection_timeout,
                                        read_timeout,
                                        max_search_results,
                                        use_multiple_search_strategies,
                                        randomize_delay,
                                        batch_progress_container  # Pass progress bar
                                    ): idx for idx, row in batch_df.iterrows()
                                }
                                
                                # Collect results as they complete
                                batch_count = 0
                                for future in concurrent.futures.as_completed(future_to_row):
                                    try:
                                        result = future.result()
                                        chunk_results.append(result)
                                        batch_count += 1
                                        
                                        # Update batch progress
                                        batch_progress_container.progress(batch_count / len(batch_df))
                                        
                                        # Update log display
                                        if batch_count % 2 == 0 or batch_count == len(batch_df):
                                            debug_log.text_area("Processing Log", log_output.getvalue(), height=150)
                                        
                                        # Save progress periodically
                                        if len(chunk_results) % auto_save_frequency == 0:
                                            current_overall_index = current_chunk * chunk_size + i + batch_count
                                            save_progress(df, results + chunk_results, current_overall_index, session_id, st.session_state['df_hash'])
                                        
                                    except Exception as e:
                                        logger.error(f"Error getting result from future: {str(e)}")
                                
                                # Update chunk progress
                                chunk_progress = (i + min(batch_size, total_rows_in_chunk - i)) / total_rows_in_chunk
                                chunk_progress_container.progress(chunk_progress)
                                
                                # Sleep to avoid rate limiting with optional randomization
                                actual_sleep = sleep_time
                                if randomize_delay:
                                    actual_sleep = sleep_time * (0.8 + 0.4 * random.random())
                                
                                logger.info(f"Completed batch. Sleeping for {actual_sleep:.2f} seconds...")
                                time.sleep(actual_sleep)
                        
                        # Chunk completed, update overall progress
                        status_text.success(f"Completed chunk {current_chunk+1}/{total_chunks}")
                        
                        # Update session state
                        st.session_state['results'] = results + chunk_results
                        st.session_state['current_chunk'] = current_chunk + 1
                        
                        # Save progress at end of chunk
                        save_progress(df, st.session_state['results'], (current_chunk + 1) * chunk_size, session_id, st.session_state['df_hash'])
                        
                        # If all chunks are done, mark as complete
                        if st.session_state['current_chunk'] >= total_chunks:
                            st.session_state['processing_complete'] = True
                            st.session_state['processing'] = False
                            
                            # Convert results to DataFrame
                            results_df = pd.DataFrame(st.session_state['results'])
                            st.session_state['results_df'] = results_df
                        
                        # Final log update
                        debug_log.text_area("Processing Log", log_output.getvalue(), height=150)
                        
                        # Rerun to either show results or process next chunk
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error during chunk processing: {str(e)}")
                        logger.error(f"Chunk processing error: {str(e)}")
                        
                        # Save progress if error occurs
                        if 'results' in st.session_state:
                            save_progress(
                                df, 
                                st.session_state['results'], 
                                st.session_state['current_chunk'] * chunk_size, 
                                session_id, 
                                st.session_state['df_hash']
                            )
                        
                        st.session_state['processing'] = False
            
            # Show results if processing is complete
            if st.session_state.get('processing_complete', False) and st.session_state.get('results_df') is not None:
                results_df = st.session_state['results_df']
                
                st.subheader("Verification Results")
                st.success(f"Completed verification of {len(results_df)} URLs!")
                
                # Highlight the Final URL column
                st.info("The 'Final URL' column contains the best URL to use - either the original valid URL or the valid alternative found through Google search.")
                
                # Filter options
                filter_option = st.radio(
                    "Filter results:",
                    ["All URLs", "Invalid URLs only", "Valid URLs only", "URLs with alternatives found"]
                )
                
                if filter_option == "Invalid URLs only":
                    filtered_df = results_df[~results_df['Is Valid']]
                elif filter_option == "Valid URLs only":
                    filtered_df = results_df[results_df['Is Valid']]
                elif filter_option == "URLs with alternatives found":
                    filtered_df = results_df[(~results_df['Is Valid']) & (results_df['Alternative URL'] != "No valid alternative found") & (~results_df['Alternative URL'].str.contains("Search error", na=False))]
                else:
                    filtered_df = results_df
                
                # Reorder columns to highlight the Final URL
                cols = filtered_df.columns.tolist()
                if 'Final URL' in cols:
                    cols.remove('Final URL')
                    # Insert Final URL right after Original URL
                    orig_idx = cols.index('Original URL') if 'Original URL' in cols else 0
                    cols.insert(orig_idx + 1, 'Final URL')
                    filtered_df = filtered_df[cols]
                
                # Display the results with pagination for very large datasets
                rows_per_page = st.slider("Rows per page", 10, 100, 50)
                
                # Pagination
                total_pages = (len(filtered_df) + rows_per_page - 1) // rows_per_page
                if 'current_page' not in st.session_state:
                    st.session_state['current_page'] = 0
                
                # Page navigation
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if st.button("Previous Page") and st.session_state['current_page'] > 0:
                        st.session_state['current_page'] -= 1
                
                with col2:
                    page_options = [f"Page {i+1} of {total_pages}" for i in range(total_pages)]
                    if page_options:
                        current_page_option = st.selectbox(
                            "Select page", 
                            options=page_options,
                            index=st.session_state['current_page']
                        )
                        st.session_state['current_page'] = page_options.index(current_page_option)
                
                with col3:
                    if st.button("Next Page") and st.session_state['current_page'] < total_pages - 1:
                        st.session_state['current_page'] += 1
                
                # Display current page of data
                start_idx = st.session_state['current_page'] * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(filtered_df))
                
                if not filtered_df.empty:
                    st.dataframe(filtered_df.iloc[start_idx:end_idx])
                    st.write(f"Showing rows {start_idx+1} to {end_idx} of {len(filtered_df)}")
                else:
                    st.warning("No data matches the current filter criteria")
                
                # Download options
                st.subheader("Download Results")
                
                # Create a "Final Results Only" DataFrame that focuses on the essential columns
                essential_cols = ['Original URL', 'Final URL', 'Is Valid', 'Alternative URL']
                other_cols = [col for col in filtered_df.columns if col not in essential_cols and col != 'Status Message']
                final_cols = essential_cols + other_cols
                final_cols = [col for col in final_cols if col in filtered_df.columns]
                
                final_results_df = filtered_df[final_cols]
                
                # Add tabs for different download options
                tab1, tab2, tab3 = st.tabs(["Full Results", "Final URLs Only", "Download Options"])
                
                with tab1:
                    if not filtered_df.empty:
                        st.dataframe(filtered_df)
                    else:
                        st.warning("No data to display")
                
                with tab2:
                    if not filtered_df.empty:
                        simple_view = final_results_df[['Original URL', 'Final URL', 'Is Valid']]
                        st.dataframe(simple_view)
                        st.info("This simplified view shows only the Original URL and the best URL to use (Final URL)")
                    else:
                        st.warning("No data to display")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    # Full results downloads
                    with col1:
                        st.subheader("Full Results")
                        if not filtered_df.empty:
                            csv = filtered_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Full Results (CSV)",
                                data=csv,
                                file_name="website_verification_results.csv",
                                mime="text/csv"
                            )
                            
                            # Create Excel file with all details
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                filtered_df.to_excel(writer, index=False, sheet_name='Full Results')
                                
                                # Add some formatting
                                workbook = writer.book
                                worksheet = writer.sheets['Full Results']
                                
                                # Add header format
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'text_wrap': True,
                                    'valign': 'top',
                                    'bg_color': '#D9E1F2',
                                    'border': 1
                                })
                                
                                # Write the column headers with the header format
                                for col_num, value in enumerate(filtered_df.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                
                                # Set column widths
                                worksheet.set_column('A:Z', 18)  # Set width for all columns
                            
                            buffer.seek(0)
                            st.download_button(
                                label="Download Full Results (Excel)",
                                data=buffer,
                                file_name="website_verification_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning("No data to download")
                    
                    # Final URLs only downloads
                    with col2:
                        st.subheader("Final URLs Only")
                        if not filtered_df.empty:
                            simple_df = filtered_df[['Original URL', 'Final URL', 'Is Valid']]
                            
                            simple_csv = simple_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Final URLs (CSV)",
                                data=simple_csv,
                                file_name="final_urls.csv",
                                mime="text/csv"
                            )
                            
                            # Create Excel file with just the essential columns
                            simple_buffer = io.BytesIO()
                            with pd.ExcelWriter(simple_buffer, engine='xlsxwriter') as writer:
                                simple_df.to_excel(writer, index=False, sheet_name='Final URLs')
                                final_results_df.to_excel(writer, index=False, sheet_name='Essential Data')
                                
                                # Add some formatting
                                workbook = writer.book
                                
                                for sheet_name in ['Final URLs', 'Essential Data']:
                                    worksheet = writer.sheets[sheet_name]
                                    
                                    # Add header format
                                    header_format = workbook.add_format({
                                        'bold': True,
                                        'text_wrap': True,
                                        'valign': 'top',
                                        'bg_color': '#D9E1F2',
                                        'border': 1
                                    })
                                    
                                    # Add conditional formatting for valid/invalid URLs
                                    valid_format = workbook.add_format({'bg_color': '#C6EFCE'})  # Light green
                                    invalid_format = workbook.add_format({'bg_color': '#FFC7CE'})  # Light red
                                    
                                    # Apply conditional formatting to the Is Valid column
                                    valid_col = 2 if sheet_name == 'Final URLs' else final_results_df.columns.get_loc('Is Valid')
                                    worksheet.conditional_format(1, valid_col, len(simple_df)+1, valid_col, 
                                                               {'type': 'cell',
                                                                'criteria': '==',
                                                                'value': True,
                                                                'format': valid_format})
                                    worksheet.conditional_format(1, valid_col, len(simple_df)+1, valid_col, 
                                                               {'type': 'cell',
                                                                'criteria': '==',
                                                                'value': False,
                                                                'format': invalid_format})
                                    
                                    # Write the column headers with the header format
                                    df_to_use = simple_df if sheet_name == 'Final URLs' else final_results_df
                                    for col_num, value in enumerate(df_to_use.columns.values):
                                        worksheet.write(0, col_num, value, header_format)
                                    
                                    # Set column widths
                                    worksheet.set_column('A:Z', 25)  # Set width for all columns
                            
                            simple_buffer.seek(0)
                            st.download_button(
                                label="Download Final URLs (Excel)",
                                data=simple_buffer,
                                file_name="final_urls.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning("No data to download")
                
                # Clear session button
                if st.button("Clear Results & Start New Verification"):
                    # Clear session state for results
                    for key in ['processing_complete', 'results_df', 'current_chunk', 'total_chunks', 
                               'results', 'df_hash', 'current_page']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Clear cache file if it exists
                    if 'df_hash' in st.session_state:
                        clear_progress(session_id, st.session_state['df_hash'])
                    
                    st.success("Results cleared! You can now start a new verification.")
                    st.experimental_rerun()
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Instructions sidebar
    with st.sidebar:
        st.subheader("Instructions")
        st.write("""
        1. Upload an Excel or CSV file containing website URLs
        2. Select the column that contains the URLs
        3. Optionally select a column with country information
        4. Adjust processing options if needed
        5. Click "Start/Resume Verification"
        6. Wait for processing to complete or use "Stop Processing" to pause
        7. Download the results when done
        """)
        
        st.subheader("About")
        st.write("""
        This app verifies if website URLs are valid and working.
        
        For invalid URLs, it attempts to find alternatives by:
        - Extracting the domain name
        - Removing TLDs and country codes
        - Adding country information (if provided)
        - Performing a Google search for the company
        - Verifying the search results
        """)
        
        st.subheader("Large File Processing")
        st.write("""
        Files are processed in chunks to prevent timeouts:
        
        - Progress is saved automatically
        - You can stop/resume at any time
        - Results are preserved between sessions
        
        For very large files (10,000+ rows):
        - Increase chunk size to reduce overhead
        - Decrease workers and batch size to reduce memory usage
        - Use longer sleep times to avoid rate limiting
        """)
        
        st.subheader("Required Packages")
        st.code("pip install streamlit pandas requests beautifulsoup4 google xlsxwriter openpyxl")

if __name__ == "__main__":
    main()
              
