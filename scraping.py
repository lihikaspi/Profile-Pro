import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for the loading bar
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
# Bright Data Proxy Configuration
# Function to initialize a WebDriver


def create_webdriver():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

# Function to get job count
def get_count(html):
    soup = BeautifulSoup(html, "html.parser")
    job_count_div = soup.find("div", class_="jobsearch-JobCountAndSortPane-jobCount")
    if job_count_div:
        count_txt = job_count_div.find("span").text.split(' ')[0]
        if len(count_txt) == 1:
            count = count_txt
        else:
            count = count_txt[:-1]
        return count
    return None

# Function to get the HTML (no retries)
def get_html(driver, job, state):
    pagination_url = 'https://www.indeed.com/jobs?q={}&l={}'
    try:
        driver.get(pagination_url.format(job, state))
        # Simulate human-like browsing behavior with random delays
        delay = random.randint(10, 12) + random.uniform(0, 2)
        time.sleep(delay)

        # Check for CAPTCHA in the page source
        html = driver.page_source
        if "captcha" in html.lower():
            # Save CAPTCHA page for debugging
            delay = random.randint(10, 15) + random.uniform(0, 2)
            time.sleep(delay)
            return None  # Return None if CAPTCHA is detected
        return html  # Return the HTML if successful
    except Exception as e:
        print(f"Error fetching data for {state}, {job}: {e}")
        return None

# Function to process each row
def process_row(state_df, state,driver, range=(0,20) ,create_flag=True):
    results = []
    if create_flag:
        driver = create_webdriver()
    state_curr_df = pd.read_csv(f'state_results/results_{state}.csv')
    try:
        for _, row in tqdm(state_curr_df.iterrows(), total=len(state_curr_df), desc=f"Processing {state}"):
            job = row['Job']
            if pd.isna(row["Count"]):
                html = get_html(driver, job, state)
                if html:
                    count = get_count(html)
                    print((state, job, count))
                    results.append((state, job, count))
                else:
                    results.append((state, job, None))
            else:
                results.append((state, job, row["Count"]))
    finally:
        if create_flag:
            driver.quit()
    return results


def combine():
    import os
    import pandas as pd

    # Folder containing the CSV files
    folder_path = "state_results"

    # Output file
    output_file = os.path.join(folder_path, "counts.csv")
    # List to store DataFrames
    dataframes = []
    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("results_") and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Read each CSV into a DataFrame
            df = pd.read_csv(file_path)
            # Add a column for the state (extracted from the filename)
            state = filename.replace("results_", "").replace(".csv", "")
            df["State"] = state
            dataframes.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"All CSVs combined and saved to {output_file}")


# List of states: put the states you want to get meta jobs counts for
states = []

# Create the DataFrame
states_df = pd.DataFrame(states, columns=["state"])
centroids_sector = [
    ('Leadership',), ('Product',), ('Engineering',), ('DataScience',), ('Operations',),
    ('Marketing',), ('Sales',), ('Design',), ('Support',), ('Finance',),
    ('Resources',), ('Research',), ('Healthcare',), ('Education',), ('Security',),
    ('Logistics',), ('Legal',), ('Quality',), ('Management',), ('Content',)
]

centroids_df = pd.DataFrame(centroids_sector, columns=['processed_title'])
states_df["key"] = 1
centroids_df["key"] = 1
df = pd.merge(states_df, centroids_df, on="key").drop("key", axis=1)

# Process each state and save results separately

"""AUTH = 'brd-customer-hl_80709a30-zone-lior_scraping:oo5wb7cq4ryx'
SBR_WEBDRIVER = 'https://brd-customer-hl_80709a30-zone-lior_scraping:oo5wb7cq4ryx@brd.superproxy.io:9515'

sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, 'goog', 'chrome')
with Remote(sbr_connection, options=ChromeOptions()) as driver:
"""
for state in states:
    # Filter rows for the current state
    state_df = df[df['state'] == state]
    print(f"Starting scraping for {state}...")
    state_results = process_row(state_df, state, '')
    print(state_results)
    # Save state-specific results
    state_results_df = pd.DataFrame(state_results, columns=["State", "Job", "Count"])
    state_results_df.to_csv(f"state_results/results_{state}.csv", index=False)
    print(f"Saved results for {state} to results_{state}.csv.")



combine()
