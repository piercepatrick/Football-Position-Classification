# Football-Position-Classification: https://lineman-classification-app.herokuapp.com/
- Created a tool that estimates whether or not a player is physically fit to play the lineman position in football (offensive or defensive) with 90% accuracy to help high school football coaches evaluate players.
- Scraped over 12,000 high school football player combine results from [NCSA Sports](https://www.ncsasports.org/football/combine-results) using BeautifulSoup. 
- Cleaned, prepared, and visualized the data using Python.
- Built baseline classification models and optimized Logistic Regression, K Nearest Neighbors, and Random Forest using RandomizedsearchCV and GridsearchCV. 
- Build a client facing API using Flask. 
- Deployed a server side API using Streamlit and Heroku.
# Define The Problem
<p> Many coaches, of all sports, face a difficult task in evaluating players at tryouts. Especially in high school, there are a wide variety of players and skillsets with a low amount of spots on the team with a variety of positions. Coaches are faced with seeing players at tryouts, which usually last just one or two days, and putting them into the best position to help the team. </p>
<p> Often times coaches will just have the baseline performance metrics for each player such as a player's 40 yard dash time, broad jump, or shuttle run. Although these drills are effective and even used as judgement in the NFL, they do not display a player's football knowledge, technique, skill, or how they match up to the opponent/other players at the position. The tool I have created allows coaches, or even the players themselves, to put in an individual's height, weight, and performance on baseline athletic drills and the tool will predict whether the player is physically comparable to other lineman. </p>
<p> The lineman position is an important one. Out of the 22 players on the field, usually about 9 or 10 of them are lineman. Lineman, both offensive and defensive, are the biggest players on the field and the best lineman excel with strength, quickness, and power. The position is often overlooked, but often times it deems the most important position group on the field. If one player makes a mistake, especially on the offensive side, the play and ultimately the game can be drastically changed. </p>
<p> This position was chosen in the use of the model because it is a position that is more about the pure physicality of a player rather than skill. For example, a coach would want to prioritze a wide receivers catching ability first, then take their physicality into consideration.  NCSA Sports did not test skill drills at their high school combines, so the lineman position had the best use case for this dataset. </p>
<img src="/Images%20for%20readme/trenches.jpg" width="500"> 

# Resources Used
- **Python Version:** 3.7 
- **Packages used:** pandas, numpy, matplotlib, sklearn, seaborn, beautifulsoup, urllib, sidetable, missingno, pickle, flask, streamlit, requests
- **Data Source:** https://www.ncsasports.org/football/combine-results 
- **Kaggle Notebook:** https://www.kaggle.com/kenjee/titanic-project-example 
- **Flask Tutorial:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2 

# Web Scraping
Utilized Python and BeautifulSoup to scrape data from each cities high school football player combine. From each cities combine data, I obtained the following: 

- Player First Name
- Player Last Name
- Player State
- Graduation Year 
- Position
- Height 
- Weight
- 40 Yard Dash Time: Fastest time recorded
- Shuttle Run Time: Tests athlete’s lateral quickness and explosion
- Three Cone Drill Time: Tests athlete’s ability to change directions at a high speed
- Broad Jump Measurement: Tests athlete’s lower body explosion and strength 
- Vertical Jump Measurement: Measures lower body explosion and power
- Player Profile Link 
- Weather Conditions on Combine Day 

<img src="/Images%20for%20readme/scrapedtable.PNG" > 

# Data Cleaning & Feature Engineering
- Combined each cities data into one DataFrame 
- Cleaned state abbreviations 
- Converted graduation year to players current school grade 
- Converted specific positions to position group (e.g. changed safeties and cornerbacks to defensive backs) 
- Cleaned and converted height data to each players height in inches 
- Created column 'profile_yn' that displays 1 if the player has a profile on the website, 0 if not 
- Created seperate columns for each weather condition 

<img src="/Images%20for%20readme/cleaning.PNG" width="500"> 

# Exploratory Data Analysis
- Displayed and visualized correlation among every feature 
- Used seaborn pairplot to see the bivariate relation between each pair of features 
- Plotted histograms and boxplots for numeric variables 
- Plotted barcharts for all categorical variables 
- Dropped the 'ATH' position, visualized and dropped null values 

<img src="/Images%20for%20readme/pairplot.PNG" width="500"> 
<img src="/Images%20for%20readme/boxplot.PNG" width="500"> 
<img src="/Images%20for%20readme/barchart.PNG" width="500"> 
<img src="/Images%20for%20readme/corr.PNG" > 

# Model Building
- Feature selection for the models
- Encoded class labels
- Standardized training data
- Implemented baseline tests of various classification models, all of which were outputting 30-40% accuracy, even after tuning.
At this point I decided to narrow down the positions to whether each player was a lineman or not, simultaneously making this a binary classification project rather than a multiclass classification.
- Models used:
  - Naive Bayes, Logistic Regression, Decision Tree, K Nearest Neighbors, Random Forest, Support Vector Classifier, XGBoost, Soft Voting Classifier 
- Highest Performing Models:
  - **Logistic Regression:** 91.9%
  - **Support Vector Classifier:** 90.7%
  - **Tuned Random Forest:** 90.7%
 
 # Local Productionization with Flask
I built a flask API endpoint by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a players stats and returns whether or not they are physically fit to play lineman.

# Web Deployment with Streamlit and Heroku
I built a web app using streamlit and deployed it online using heroku. 

<img src="/Images%20for%20readme/webapp.PNG"> 
