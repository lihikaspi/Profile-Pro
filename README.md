<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Profile Pro: A LinkedIn Profile Optimizer</h1>
<h3 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Created By: Lihi Kaspi, Harel Oved & Lior Zaphir</h3>
<br>

<p align="center">
  <img src="Photos/linkedin_photo.png" alt="Logo" width="300" height="300">

# Contents
- [Overview](#Overview)
- [Running The Code](#Running-The-Code)
  - ["Meta-Job" Classification](##"Meta-Job"-Classification)
  - [Scraping](##Scraping)
  - [Profile Score Calculation](##Profile-Score-Calculation)
  - [Data Preprocessing](##Data-Preprocessing)
  - ["about" Section Optimization](##"about"-Section-Optimization)
  - [Profile Pro](##Profile-Pro) 


# Overview
The Project aims to help you optimize your LinkedIn profile <br> 
We provide a service that allows users to monitor their profiles' success using AI-Powered tools.
Our service helps your profile stand out and  


# Running The Code
The project contains **TBD** python files and **TBD** Jupyter notebooks that should be ran in this specified order since files saved by one notebook are needed for the next one. <br>
For each notebook description we added the cells numbers where you'll need to change the file paths in order to run the code.

## "Meta-Job" Classification
The code takes the job titles and positions from the user profiles and classifies them into 20 "meta-jobs" to be used for the scarping process.

**File paths to change:** <br>
Input - cell 2: path to the profiles dataset <br> 
Output - cell 15: path to the the parquet file containing each job in the dataset and its "meta-job"

## Scraping
The code takes the 20 "meta-jobs" created in the "Metajobs_Classification" notebook and scrapes the job search engine website *indeed.com* for the number of result you get by searching every pair of meta job and location 

**To run the code:** IDK

## Profile Score Calculation
The code calculates the Profile Score for user profiles based on existing column and poplurity of the field they work in.
At the end, the code saves two parquet files (one for users and one for companies) with all the original columns of the profiles and a new column 'profile_score'

**File paths to change:** <br>
Input - cell 3: paths to the profiles dataset, to the job clasifications parquet saved in the "Metajobs classification" notebook, and the scraped data.  <br> 
Output - cell 18: path to the the parquet file containing the original profiles dataset and an additional column of the numeric profile score

## Data Preprocessing
The code takes the dataframe containing the profile scores and prepares a new file containing a features vector column to put into the classification model.

**File paths to change:** <br>
Input - cell 2: path to the profile scores parquet saved in the "profiles_score_calculation" notebook.  <br> 
Output - cell **TBD**: path to the the parquet file containing the features vectors

## "about" Section Optimization
IDK

## Profile Pro
The code trains and evaluates the score classification model, offers suggestions for the bad profiles and re-evaluates the changed and optimized profiles profiles

**File paths to change:** <br>
Input - cell 4: path to the profiles dataset, to the preprocessed data saved in the "data_preprocessing" notebook, and to the new and optimized "about" sections. <br> 
Output - cell **TBD**: path to the the parquet file containing the features vectors

