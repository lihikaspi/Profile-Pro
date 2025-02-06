<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Profile Pro: A LinkedIn Profile Optimizer</h1>
<h3 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Created By: Lihi Kaspi, Harel Oved & Lior Zaphir</h3>
<br>

<p align="center">
  <img src="Photos/linkedin_photo.png" alt="Logo" width="300" height="300">

# Contents
- [Overview](#Overview)
- [Running The Code](#Running-The-Code)
  - [Scraping](#Scraping)
  - ["Meta-Job" Classification](#"Meta-Job"-Classification)
  - [Profile Score Calculation](#Profile-Score-Calculation)
  - [Data Preprocessing](#Data-Preprocessing)
  - [LLM Prompt Engineering](#LLM-Promp-Engineering)
  - [Profile Pro](#Profile-Pro) 


# Overview
Our project focuses on how to improve a LinkedIn profile to appeal to more people and, specifically, companies, job recruiters, and future investors. <br>
We provide a service that allows users to monitor their profiles' success using AI-Powered tools. <br>
*Profile Pro* is a tool that analyzes an existing LinkedIn profile and suggests changes that could make the profile more informative about the person behind it by suggesting changes to various fields in the profile.


# Running The Code
The project contains a python file for the scraping process and 5 Jupyter notebooks that should be run in this specified order since files saved by one notebook are needed for the next one. <br>
For each notebook description we added the cells numbers where you'll need to change the file paths in order to run the code.

## Scraping
The code takes 20 "meta-jobs" chosen to reresent different fields of the workforce and scrapes the job search engine website *indeed.com* for the number of result you get by searching every pair of meta job and US state. 

**To run the code: OPTIONAL** <br>
Saving the output - lines 64, 89 and 147 contain the paths to save the scraped data, the files will be saved in the local folder of the code file.

## "Meta-Job" Classification
The code takes the job titles and positions from the profiles and classifies them into the 20 "meta-jobs" used for the scraping process.

**File paths to change:** <br>
Input - cell 2: path to the profiles dataset <br> 
Output - cell 12: path to the the parquet file containing each job in the dataset and its "meta-job"

## Profile Score Calculation
The code calculates the Profile Score for each profile based on the existing columns and poplurity of the field they work in (the scraped data).

**File paths to change:** <br>
Input - cell 3: paths to the profiles dataset, to the job clasifications parquet saved in the "Metajobs_classification" notebook, and the csv of the scraped data.  <br> 
Output - cell 18: path to the the parquet file containing the original profiles dataset and an additional column of the numeric profile score

## Data Preprocessing
The code takes the dataset containing the profile scores and prepares a new file containing a features vector column to put through the classification model.

**File paths to change:** <br>
Input - cell 2: path to the profile scores parquet saved in the "profiles_score_calculation" notebook.  <br> 
Output - cell 8: path to the the parquet file containing profile ids, the features vectors and the real scores.

## LLM Promp Engineering
...

**File paths to change:** <br>
Input - cell 2: path to the profile scores parquet saved in the "profiles_score_calculation" notebook.  <br> 
Output - cell IDFK?? : ???????????????

## Profile Pro
The code trains and evaluates the score classification model, offers suggestions for the bad profiles and re-evaluates the changed and optimized profiles profiles

**File paths to change:** <br>
Input - cell 4: path to the profiles dataset and to the preprocessed data saved in the "data_preprocessing" notebook <br>
Saving and loading the classification model - cells 12 and 17: path to save the trained multilayer perceptron model <br>
Output - cell **26 PROBABLY**: path to the the parquet file containing the suggestion for each user.

