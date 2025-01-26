# Profile Pro: A LinkedIn Profile Optimizer
### Created By: Lihi Kaspi, Harel Oved & Lior Zaphir
<br>
--TODO: find photo

## Contents
- [Overview](##Overview)
- [Running the Code](##Running-the-Code)
  - [Scraping](###Scraping) --TODO: change to real name
  - [Data Preprocessing](###Data-Preprocessing)
  - [Profile Score Calculation](###Profile-Score-Calculation)
  - [Project](###Project) --TODO: change to more informative name



## Overview


## Running the Code
### Scraping


### Profile Score Calculation
The code calculates the Profile Score for User Profiles and Company Profiles based on existign column and poplurity of the field they work in.
At the end, the code saves two parquet files (one for users and one for companies) with all the original columns of the profiles and a new column 'profile_score'

**To run the code:** change file paths in cells **11** and **TBD** to desired paths to save new datasets of users and companies containing the respective scores

### Data Preprocessing
The code takes the dataframe containing the profile score and prepares a new vector column to train the model on

**To run the code:** 
1. change file path in cell **2** to the path where you saved the profiles_with_scores.parquet file from the "Profile_Score_Calculation" notebook
2. change file path in cell **12** to desired path to save the processed data


### Project
The code trains and evaluates the score-predicting model and then offers suggestion for the bad profiles

**To run the code:** change the file paths in cell **TBD** to the path where you saved the processed dataset from the "Data_Preprocessing" notebook

