import streamlit as st
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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Website Verifier", layout="wide")

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
            import socket
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

# Function to clean and parse URL domain
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
    
    st.title("Website URL Verifier")
    st.write("Upload a spreadsheet with website URLs to verify and find alternatives for invalid ones.")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Load the data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("File uploaded successfully!")
            st.write(f"Found {len(df)} rows and {len(df.columns)} columns")
            
            # Display preview
            st.subheader("Preview of the data")
            st.dataframe(df.head())
            
            # Select URL column
            url_column = st.selectbox(
                "Select the column containing website URLs",
                options=df.columns.tolist()
            )
            
            # Select Country column (optional)
            col1, col2 = st.columns(2)
            with col1:
                use_country = st.checkbox("Use country information to improve search", value=True)
            
            country_column = None
            if use_country:
                with col2:
                    country_options = ["None"] + df.columns.tolist()
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
            
            # Advanced options
            with st.expander("Advanced Options"):
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
            
            if st.button("Start Verification"):
                # Initialize results list
                results = []
                
                # Set up progress bar
                total_rows = len(df)
                progress_text = "Verifying URLs. Please wait..."
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Create a debug log area
                debug_log = st.empty()
                
                # Setup a handler to show logs in the Streamlit app
                log_output = io.StringIO()
                log_handler = logging.StreamHandler(log_output)
                log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(log_handler)
                
                # Process in batches with parallel execution
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        for i in range(0, total_rows, batch_size):
                            batch_df = df.iloc[i:min(i+batch_size, total_rows)]
                            
                            # Update status
                            current_batch = f"{i+1} to {min(i+batch_size, total_rows)}"
                            status_text.text(f"{progress_text} Processing rows {current_batch} of {total_rows}")
                            logger.info(f"Processing batch: rows {current_batch} of {total_rows}")
                            
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
                                    randomize_delay
                                ): idx for idx, row in batch_df.iterrows()
                            }
                            
                            # Collect results as they complete
                            for future in concurrent.futures.as_completed(future_to_row):
                                try:
                                    result = future.result()
                                    results.append(result)
                                    
                                    # Update progress
                                    current_progress = len(results) / total_rows
                                    progress_bar.progress(current_progress)
                                    
                                    # Update log display periodically
                                    if len(results) % 5 == 0 or len(results) == total_rows:
                                        debug_log.text_area("Processing Log", log_output.getvalue(), height=150)
                                        
                                except Exception as e:
                                    logger.error(f"Error getting result from future: {str(e)}")
                            
                            # Sleep to avoid rate limiting with optional randomization
                            actual_sleep = sleep_time
                            if randomize_delay:
                                actual_sleep = sleep_time * (0.8 + 0.4 * random.random())  # 80% to 120% of sleep_time
                            
                            logger.info(f"Completed batch. Sleeping for {actual_sleep:.2f} seconds...")
                            time.sleep(actual_sleep)
                
                except Exception as e:
                    st.error(f"Error during batch processing: {str(e)}")
                    logger.error(f"Batch processing error: {str(e)}")
                
                # Final log update
                debug_log.text_area("Processing Log", log_output.getvalue(), height=150)
                
                # Create and display results DataFrame
                results_df = pd.DataFrame(results)
                st.success(f"Completed verification of {len(results_df)} URLs!")
                
                # Display results
                st.subheader("Verification Results")
                
                # Highlight the Final URL column for clarity
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
                
                st.dataframe(filtered_df)
                
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
                    st.dataframe(filtered_df)
                
                with tab2:
                    st.dataframe(final_results_df[['Original URL', 'Final URL', 'Is Valid']])
                    st.info("This simplified view shows only the Original URL and the best URL to use (Final URL)")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    # Full results downloads
                    with col1:
                        st.subheader("Full Results")
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
                    
                    # Final URLs only downloads
                    with col2:
                        st.subheader("Final URLs Only")
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
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Instructions sidebar
    with st.sidebar:
        st.subheader("Instructions")
        st.write("""
        1. Upload an Excel or CSV file containing website URLs
        2. Select the column that contains the URLs
        3. Optionally select a column with country information
        4. Adjust batch size and worker count if needed
        5. Click "Start Verification"
        6. Wait for processing to complete
        7. Download the results
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
        
        st.subheader("How Country Information Helps")
        st.write("""
        Including country information significantly improves search results by:
        - Focusing the search on the correct geographic region
        - Identifying country-specific domains or websites
        - Resolving ambiguities for companies with similar names in different countries
        - Finding local branches of international companies
        """)
        
        st.write("Note: To use Google search functionality, you need to install the google package:")
        st.code("pip install google")

if __name__ == "__main__":
    main()
