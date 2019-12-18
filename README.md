# Data Science Final Project
Charles Laurent, AJ Marino, Jenny Park, 

## Setup
1. Download College Scorecard data. 
2. Replace `datadir` with location where college data csv files are located
3. Jupyter Notebooks can be run separately using the `collegedatahelper.py` library

## Datasets

* College Scorecard Data: https://collegescorecard.ed.gov/data/
* GDP Data: https://www-statista-com.proxy.library.nyu.edu/statistics/188105/annual-gdp-of-the-united-states-since-1990/
* State Abbreviations: http://worldpopulationreview.com/states/state-abbreviations/
* Median Household Income by State: https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk


## File Structure

    .
    ├── 00-sandbox.ipynb                        # Sandbox
    ├── 01-a-random-forest-regressor.ipynb      # Random Forest Regressor using 1 year of data
    ├── 01-b-box-plots.ipynb                    # Box Plots
    ├── 01-random-forest-full.ipynb             # Random Forest Regressor using all 22 years of data
    ├── 02-decision-tree.ipynb                  # Decison Tree Generator
    ├── 03-external-data.ipynb                  # Adding external data (median household income by state data)
    ├── 03-gdp-full.ipynb                       # Adding external data (GDP) - full dataset
    ├── collegedatahelper.py                    # Helper file that performs data cleanup
    └── README.md
