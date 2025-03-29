import pandas as pd
import os

root = '../In/Box_data/'
for file in os.listdir('../In/Box_data/'): 
       
        if file.endswith("BLN.xlsx") or file.endswith("6WK.xlsx"):
            print('Read: ' + root + file)
            # file_n = root + file[0:7] + '_seg_pos.csv'          
            # segment_pos = pd.read_excel(root + file, sheet_name='Segment Position') # , skiprows=[0], usecols="B:BR"
            # segment_pos.to_csv(file_n, header=True, index=False) # 
                
            file_n = root + file[0:7] + '_seg_qua.csv'          
            segment_qua = pd.read_excel(root + file, sheet_name='Segment Orientation - Quat') # , skiprows=[0], usecols="B:BR"
            segment_qua.to_csv(file_n, header=True, index=False) #   

            file_n = root + file[0:7] + '_seg_eul.csv'          
            segment_eul = pd.read_excel(root + file, sheet_name='Segment Orientation - Euler') # , skiprows=[0], usecols="B:BR"
            segment_eul.to_csv(file_n, header=True, index=False) #         

            # file_n = root + file[0:7] + '_seg_acc.csv'          
            # segment_acc = pd.read_excel(root + file, sheet_name='Sensor Free Acceleration') # , skiprows=[0], usecols="B:BR"
            # segment_acc.to_csv(file_n, header=True, index=False) #   

            # file_n = root + file[0:7] + '_seg_ang.csv'          
            # segment_acc = pd.read_excel(root + file, sheet_name='Ergonomic Joint Angles XZY') # , skiprows=[0], usecols="B:BR"
            # segment_acc.to_csv(file_n, header=True, index=False) #             

            file_n = root + file[0:7] + '_sen_eul.csv'          
            segment_acc = pd.read_excel(root + file, sheet_name='Sensor Orientation - Euler') # , skiprows=[0], usecols="B:BR"
            segment_acc.to_csv(file_n, header=True, index=False) #    

            file_n = root + file[0:7] + '_sen_qua.csv'          
            segment_acc = pd.read_excel(root + file, sheet_name='Sensor Orientation - Quat') # , skiprows=[0], usecols="B:BR"
            segment_acc.to_csv(file_n, header=True, index=False) #          

            print('Split: ' + root + file) 
                     
                     
print("---- .xlsx to .csv complete ----")