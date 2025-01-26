<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Profile Pro: A LinkedIn Profile Optimizer</h1>
<h3 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Created By: Lihi Kaspi, Harel Oved & Lior Zaphir</h3>
<br>
--TODO: find photo

<p align="center">
  <img src="Photos/linkedin_photo.png" alt="Logo" width="300" height="300">

## Contents
- [Overview](##Overview)
- [Running The Code](##Running-The-Code)
  - [Scraping](###Scraping) --TODO: change to real name
  - [Data Preprocessing](###Data-Preprocessing)
  - [Profile Score Calculation](###Profile-Score-Calculation)
  - [Profile Pro](###Profile-Pro) 



## Overview
The Project aims to help you optimize your LinkedIn profile ...

## Running The Code
The project contains 4 Jupyter notebooks that should be ran in this specified order since files saved by one notebook are needed for the next one. <br>
For each notebook description we added the cells numbers where you'll need to change the file paths in order to run the code.

### Scraping
The code scrapes the website *indeed.com* for the number of result you get by searching a job and location 

**To run the code:** change file path in cell **TBD** to desired path to save the scraped data


### Profile Score Calculation
The code calculates the Profile Score for User Profiles and Company Profiles based on existign column and poplurity of the field they work in.
At the end, the code saves two parquet files (one for users and one for companies) with all the original columns of the profiles and a new column 'profile_score'

**To run the code:** 
1. change file paths in cell **4** to the paths were the profiles and companies datasets and scraped sata you saved in the "Scraping" notebook --TODO: change to real name 
2. change file paths in cells **11** and **TBD** to desired paths to save new datasets of users and companies containing the respective scores

### Data Preprocessing
The code takes the dataframe containing the profile score and prepares a new vector column to train the model on

**To run the code:** 
1. change file path in cell **2** to the path where you saved the profiles_with_scores.parquet file from the "Profile_Score_Calculation" notebook
2. change file path in cell **12** to desired path to save the processed data


### Profile Pro
The code trains and evaluates the score-predicting model and then offers suggestion for the bad profiles

**To run the code:** change the file paths in cell **TBD** to the path where you saved the processed dataset from the "Data_Preprocessing" notebook

