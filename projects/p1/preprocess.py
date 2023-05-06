import pandas as pd
from feature_engine import encoding, imputation 
from sklearn import base, pipeline
import numpy as np

def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    return (df_
            .assign(age=df_.Q2.str.slice(0, 2).astype(int),
                    education=df_.Q4.replace({
                        "Master's degree": 18,
                        "Bachelor's degree": 26,
                        "Doctoral degree":20,
                        "Some college/university study without earning a bachelor's degree": 13,
                        "Professional degree": 19,
                        "I prefer not to answer": None,
                        "No formal education past high school": 12
                    }),
                    major=(df_.Q5
                           .pipe(topn, n=3)
                           .replace({
                               "Computer science (software engineering etc.)" : 'cs',
                               "Engineering (non-computer focused)" : 'eng',
                               "Mathermatics or statistics": "stat"})
                           ),
                    years_exp = (df_.Q8.str.replace("+", "", regex=False)
                                 .str.split("-", expand=True)
                                 .iloc[:,0]
                                 .astype(float)),
                    
                    compensation=(df_.Q9.str.replace("+", "", regex=False)
                                  .str.replace(",","", regex=False)
                                  .str.replace("500000", "500", regex=False)
                                  .str.replace("I do not wish to disclose my approximate yearly compensation", "0", regex=False)
                                  .str.split("-", expand=True)
                                  .iloc[:,0]
                                  .fillna(0)
                                  .astype(int)    
                                  .mul(1_000)),
                    python=df_.Q16_Part_1.fillna(0).replace("Python", 1),
                    r = df_.Q16_Part_2.fillna(0).replace("R", 1),
                    sql=df_.Q16_Part_3.fillna(0).replace("SQL", 1)
                    )
                    .rename(columns=lambda col:col.replace(" ","_"))
                    .loc[:, "Q1,Q3,age,education,major,years_exp,compensation,python,r,sql".split(",")]
    )

def topn(ser, n=5, default="other"):
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)

