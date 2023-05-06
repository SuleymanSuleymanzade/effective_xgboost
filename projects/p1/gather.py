import pandas as pd 
import urllib.request 
import zipfile 

url = "https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
fname = "kaggle-survey-2028.zip"
member_name = "multipleChoiceResponses.csv"

def extract_zip(src, dst, member_name):
    url = src 
    fname = dst 
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, "wb") as fout:
        fout.write(data)
    
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw 
    
raw = extract_zip(url, fname, member_name)
print(raw)