# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:33:38 2015

@author: zhihuixie
"""

import pandas as pd
import numpy as np


class Feature_engineering():
    """
    this class includes function to handle missing values and outlier 
    (clean_data), and extract new features (extract_date_feature, 
    extract_other_feature)
    """
    def __init__(self, df):
        # input pandas data frame as a parameter
        self.df = df
    def replace_outlier(self, data, thresh=3.5):
        """
        replace outlier for numeric data
        References:
        ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        nrows = len(data)
        median = np.median(data)
        diff = np.abs(data - median)
        mdev = np.median(diff)
        modified_z_score = 0.6745 *  diff/mdev if mdev else [0 for i in \
                           range (diff.shape[0])]
        for j in range(nrows):
            #replaace values that z score is larger than thresh
            if modified_z_score[j] > thresh: 
                data[j] = median
        return data        
    
    def clean_data(self):
        """
        this function handles missing values for numeric and categorical data
        and replace outlier for numeric data
        """
        # replace column names
        self.df.columns = ["permalink", "name", "homepage_url", "category_list"\
                           , "market","funding_total_usd", "status", \
                           "country_code", "state_code","region", "city", \
                           "funding_rounds", "founded_at", "founded_month",\
                           "founded_quarter", "founded_year", "first_funding_at",\
                           "last_funding_at"]
        # remove data that founded year at 2014 because the data in this year 
        # are incompleted
        self.df = self.df[self.df.founded_year < 2014]
        print "handle missing values and transform..."
        count_null = list(self.df.isnull().sum())
        (nrows, ncols) = self.df.shape
        for i in range(ncols):
            # drop columns with nan values over 80%
            if count_null[i] > 0.8* nrows:
                self.df.drop(self.df.columns[i], axis = 1, inplace = True) 
                print "drop column with missing values over 80 percent: column %d"%i
            # re-count number of missing values after dropping columns
        count_null = list(self.df.isnull().sum())
        (nrows, ncols) = self.df.shape
        for j in range(ncols):
            print "treat column %d..."%j
            # handle categorical missing values
            if self.df.iloc[:, j].dtypes == "object":
                if count_null[j] != 0:
                    self.df.iloc[:, j].fillna(" ", inplace = True)
            # replace missing numeric value and outlier with median
            else:
                if count_null[j] != 0:
                    self.df.iloc[:, j].fillna(-1, inplace = True)
                data = list(self.df.iloc[:,j])
                replaced_data = self.replace_outlier(data, thresh=3.5)
                print "Is the train data column %d replaced?"%j, \
                   data != replaced_data
                self.df.iloc[:,j] = replaced_data
            print "process column %d, completed" %j 
            
    def clean_strings(self):
        """
        replace dummy characters or meaningness words in a string in features
        """
        treat_columns = ["permalink", "name", "homepage_url", "category_list"]
        chars = ["/", "#", "-", ".", "|", ":", "organization", "http", "www"]
        for feature in treat_columns:
            for ch in chars:
                self.df[feature] = self.df[feature].str.replace(ch, " ")

    def extract_date_features(self):
        """
        get date information: year, month and day
        """
        self.df["founded_year"] = self.df.founded_at.apply(lambda x: x[:4])
        self.df["founded_month"] = self.df.founded_at.apply(lambda x: x[5:7])
        self.df["founded_day"] = self.df.founded_at.apply(lambda x: x[-2:])
        self.df["founded_quarter"] = self.df.founded_quarter.apply(lambda x: x[-2:])
        # drop date columns after extraction
        self.df.drop("founded_at", axis = 1, inplace = True)
        self.df.drop("first_funding_at", axis = 1, inplace = True)
        self.df.drop("last_funding_at", axis = 1, inplace = True)
        
    def convert_string_to_float(self, s):
        """
        remove dummy chacters
        """
        s = s.replace(" ", "")
        s = s.replace("-", "0")
        s = s.replace(",", "")
        return s
            
    def extract_other_features(self):
        """
        convert string type of total funding feature as float
        and calculate averaged funding per round
        """
        self.df["funding_total_usd"] = self.df.funding_total_usd.apply\
                      (lambda x: "".join(map(self.convert_string_to_float, x)))
        self.df.funding_total_usd = self.df.funding_total_usd.astype(float)
        self.df["funding_per_round"] = self.df.funding_total_usd/self.df.funding_rounds
                
    def output_to_file(self):
        """
        write cleaned data and extracted features to file
        """
        self.clean_data()
        self.clean_strings()
        self.extract_date_features()
        self.extract_other_features()
        self.df.to_csv("cleaned startups data_2013.csv", index = False)
        

if __name__ == "__main__":
    print "load data..."
    df = pd.read_csv("crunchbase_monthly_export_companies_2014.csv")
    print "process data..."
    feature_process = Feature_engineering(df)
    feature_process.output_to_file()
    print "processing completed!!!"
    
