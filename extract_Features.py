import data_creation_v3
import datetime
import math
import pandas as pd
import numpy as np
import whois
from tqdm import tqdm
from interruptingcow import timeout
import os
print(os.getcwd())
os.chdir(r'C:\Users\abbya\OneDrive\Desktop\simply\URL')

l = ['Defacement.csv','Phishing.csv','Malware.csv','Spam.csv','Benign.csv']

desired_columns = ['File', 'bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps', 'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams', 'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars', 'sscr', 'urlIsLive', 'urlLength','has_tld','tld_count','cwsc','NS_count','MX_count','has_ssl','redirect_count','asn','ipgeo','ptr','has_rbl','blacklisted']

emp = data_creation_v3.UrlFeaturizer("").run().keys()
A = pd.DataFrame(columns = desired_columns)
t=[]
for j in l:
    print(j)
    d=pd.read_csv(j,header=None).to_numpy().flatten()
    for i in tqdm(d):
        # try:
        #     with timeout(30, exception = RuntimeError):
                temp=data_creation_v3.UrlFeaturizer(i).run()
                temp.pop("ext",None)
                temp["File"]=j.split(".")[0]
                t.append(temp)
        # except RuntimeError:
        #     pass

# Convert each dictionary in t to a DataFrame
t_dataframes = [pd.DataFrame.from_dict(entry, orient='index').T for entry in t]

# Concatenate the list of DataFrames with A
A = pd.concat([A] + t_dataframes, ignore_index=True)
os.chdir(r'C:\Users\abbya\OneDrive\Desktop\simply')
A.to_csv("featuresssssss.csv")