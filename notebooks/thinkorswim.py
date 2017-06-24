import os, sys
import pandas
import seaborn as sns
import csv
import numpy as np
import datetime 
import pandas  

def extract_option_info(option_string,opt):
    split = option_string.split()
    if opt == "Expiry":
        return pandas.to_datetime(split[0]+' '+split[1]+' '+split[2])
    elif opt == "Strike":
        return float(split[3])
    elif opt == "Type":
        return split[4]
    else:
        return None
    
def update_option(row,opt=""):
        return extract_option_info(row['Option'],opt)
    
def initialize_dataframe(df):
    df["Expiry"] = df.apply(lambda row: update_option(row,"Expiry"),axis=1)
    df["Strike"] = df.apply(lambda row: update_option(row,"Strike"),axis=1)
    df["Type"] = df.apply(lambda row: update_option(row,"Type"),axis=1) 
    df.set_index('Expiry', append=True, inplace=True)
    df.set_index('Strike', append=True, inplace=True)
    df.set_index('Type', append=True, inplace=True)
    df = df.reset_index(level='Time')
    df.set_index('Time', append=True, inplace=True)
    df = df.sort_index(level=0)
    return df
    #.sort_index(level=0)
    
    
def check_file_for_header(path):
    newfile = None
    for i,line in enumerate(open(path)):
        if i == 0 and "Time" in line:
            break
        else:
            if i == 0:
                newfile = open(path+'.tmp','w')
                newfile.write('Time	Option	Qty	Price	Exchange	Market	Delta	IV	Underlying	Condition	\n')
            newfile.write(line)
    if newfile != None:
        newfile.close()
        os.rename(path,path+'.old')
        os.rename(path+'.tmp',path)
        
        
def load_tos_datafile(path): 
    check_file_for_header(path)
    data = pandas.DataFrame.from_csv(path,sep='\t')
    data.drop(data.columns[9],1,inplace=True)
    data = initialize_dataframe(data)
    return data

