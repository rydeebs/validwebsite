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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
        
        # Check if response is a success (200 OK)
        if response.status_code == 200:
            # Check if the page is not a common error page
            content = response.text.lower()
            if any(term in content for term in ['404 not found', 'page not found', 'site not found', 'access denied']):
                return False, f"Page content indicates error: Status code {response.status_code}"
            return True, response.url
        else:
            return False, f"HTTP Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)

# Function to clean and parse URL domain
def get_search_query(url, country=None):
    # Remove http://, https://, www.
    domain = urlparse(url).netloc if urlparse(url).netloc else url
    domain = domain.replace('www.', '')
    
    # Remove common TLDs and country codes
    tld_pattern = r'\.(com|net|org|co|io|gov|edu|uk|us|ca|au|de|fr|jp|cn|in|br).*$'
    domain = re.sub(tld_pattern, '', domain)
    
    # Replace symbols with spaces
    domain = re.sub(r'[-_]', ' ', domain)
    
    # Build the search query
    search_query = domain + " factory manufacturing supplier"
    
    # Add country to the search query if provided
    if country and isinstance(country, str) and len(country.strip()) > 0:
        # Clean up country name
        country = country.strip()
        # Add country to search query
        search_query += f" {country}"
    
    return search_query

# Function to find alternative URL via Google search
def find_alternative_url(url, country=None):
    search_query = get_search_query(url, country)
    try:
        st.session_state['last_search_query'] = search_query  # Store for debugging
        
        for result in search(search_query, num=5, stop=5, pause=2):
            # Verify if the result is a valid website
            is_valid, result_url = check_url(result)
            if is_valid:
                return result_url
    except Exception as e:
        return f"Search error: {str(e)}"
    
    return "No valid alternative found"

# Function to process a single row
def process_row(row_data, url_column, country_column=None, progress_bar=None):
    url = row_data[url_column]
    result = {}
    
    # Copy all original data
    for col in row_data.index:
        result[col] = row_data[col]
    
    # Get country if country column is provided
    country = None
    if country_column and country_column in row_data:
        country = row_data[country_column]
    
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
        alternative_url = find_alternative_url(url, country)
        result['Alternative URL'] = alternative_url
        result['Final URL'] = alternative_url if alternative_url and not alternative_url.startswith("Search error") and not alternative_url == "No valid alternative found" else url
    else:
        result['Alternative URL'] = ""
        result['Final URL'] = url
    
    # Update progress bar if provided
    if progress_bar is not None:
        progress_bar.progress(1)
    
    return result

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
                                      min_value=1, max_value=20, value=5)
            with col2:
                max_workers = st.slider("Number of parallel workers", 
                                       min_value=1, max_value=10, value=3)
            
            # Advanced options
            with st.expander("Advanced Options"):
                st.markdown("### Search Enhancement")
                include_industry = st.checkbox("Include industry-specific terms in search", value=True)
                if include_industry:
                    industry_terms = st.text_input(
                        "Industry-specific search terms (comma separated)",
                        value="factory,manufacturing,supplier,producer"
                    )
                st.markdown("### Rate Limiting Protection")
                sleep_time = st.slider("Sleep time between batches (seconds)", 
                                       min_value=1, max_value=10, value=2)
            
            if st.button("Start Verification"):
                # Initialize results list
                results = []
                
                # Set up progress bar
                total_rows = len(df)
                progress_text = "Verifying URLs. Please wait..."
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Process in batches with parallel execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(0, total_rows, batch_size):
                        batch_df = df.iloc[i:min(i+batch_size, total_rows)]
                        
                        # Update status
                        status_text.text(f"{progress_text} Processing rows {i+1} to {min(i+batch_size, total_rows)} of {total_rows}")
                        
                        # Submit batch for processing
                        future_to_row = {executor.submit(process_row, row, url_column, country_column): idx 
                                         for idx, row in batch_df.iterrows()}
                        
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(future_to_row):
                            results.append(future.result())
                            
                            # Update progress
                            progress_bar.progress(len(results) / total_rows)
                        
                        # Sleep to avoid rate limiting
                        time.sleep(sleep_time)
                
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
, '', domain)
    
    # Replace symbols with spaces
    domain = re.sub(r'[-_]', ' ', domain)
    
    # Build the search query
    search_query = domain + " factory manufacturing supplier"
    
    # Add country to the search query if provided
    if country and isinstance(country, str) and len(country.strip()) > 0:
        # Clean up country name
        country = country.strip()
        # Add country to search query
        search_query += f" {country}"
    
    return search_query

# Function to find alternative URL via Google search
def find_alternative_url(url):
    search_query = get_search_query(url)
    try:
        for result in search(search_query, num=5, stop=5, pause=2):
            # Verify if the result is a valid website
            is_valid, result_url = check_url(result)
            if is_valid:
                return result_url
    except Exception as e:
        return f"Search error: {str(e)}"
    
    return "No valid alternative found"

# Function to process a single row
def process_row(row_data, url_column, progress_bar=None):
    url = row_data[url_column]
    result = {}
    
    # Copy all original data
    for col in row_data.index:
        result[col] = row_data[col]
    
    # Check if URL is valid
    is_valid, message = check_url(url)
    result['Original URL'] = url
    result['Is Valid'] = is_valid
    result['Status Message'] = message
    
    # If not valid, find alternative
    if not is_valid:
        alternative_url = find_alternative_url(url)
        result['Alternative URL'] = alternative_url
        result['Final URL'] = alternative_url if alternative_url and not alternative_url.startswith("Search error") and not alternative_url == "No valid alternative found" else url
    else:
        result['Alternative URL'] = ""
        result['Final URL'] = url
    
    # Update progress bar if provided
    if progress_bar is not None:
        progress_bar.progress(1)
    
    return result

def main():
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
            url_columns = st.selectbox(
                "Select the column containing website URLs",
                options=df.columns.tolist()
            )
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("Batch size (higher may be faster but could hit rate limits)", 
                                      min_value=1, max_value=20, value=5)
            with col2:
                max_workers = st.slider("Number of parallel workers", 
                                       min_value=1, max_value=10, value=3)
            
            if st.button("Start Verification"):
                # Initialize results list
                results = []
                
                # Set up progress bar
                total_rows = len(df)
                progress_text = "Verifying URLs. Please wait..."
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Process in batches with parallel execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(0, total_rows, batch_size):
                        batch_df = df.iloc[i:min(i+batch_size, total_rows)]
                        
                        # Update status
                        status_text.text(f"{progress_text} Processing rows {i+1} to {min(i+batch_size, total_rows)} of {total_rows}")
                        
                        # Submit batch for processing
                        future_to_row = {executor.submit(process_row, row, url_columns): idx 
                                         for idx, row in batch_df.iterrows()}
                        
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(future_to_row):
                            results.append(future.result())
                            
                            # Update progress
                            progress_bar.progress(len(results) / total_rows)
                        
                        # Sleep to avoid rate limiting
                        time.sleep(1)
                
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
        3. Adjust batch size and worker count if needed
        4. Click "Start Verification"
        5. Wait for processing to complete
        6. Download the results
        """)
        
        st.subheader("About")
        st.write("""
        This app verifies if website URLs are valid and working.
        
        For invalid URLs, it attempts to find alternatives by:
        - Extracting the domain name
        - Removing TLDs and country codes
        - Performing a Google search for the company
        - Verifying the search results
        """)
        
        st.write("Note: To use Google search functionality, you need to install the google package:")
        st.code("pip install google")

if __name__ == "__main__":
    main()
