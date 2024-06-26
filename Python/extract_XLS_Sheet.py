import pandas as pd
import os


for (root,dirs,files) in os.walk('../In/Box_data/', topdown=True): 
        for file in files:
                if file.endswith(".xlsx"):
                    file_n = root + file[0:6] + '_seg_pos.csv'          
                    segment_pos = pd.read_excel(root + file, sheet_name='Segment Position') # , skiprows=[0], usecols="B:BR"
                    segment_pos.to_csv(file_n, header=True, index=False) # 
                    print(file_n)  
                     
                     
print("---- .xlsx to .csv complete ----")