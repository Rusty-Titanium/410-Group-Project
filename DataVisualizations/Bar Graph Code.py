import pandas as pd
import os
import matplotlib.pyplot as plt

path = '/Users/jasonbradley/Downloads/410-Group-Project-main/Stocks By Sector'

sector_dict = {'BHP':'Basic Material' , 'FAST':'Basic Material', 'GFI':'Basic Material', 'GPK':'Basic Material', 'HL':'Basic Material', 'PGK':'Basic Material',
                'RIO':'Basic Material', 'SUZ':'Basic Material', 'AIN':'Consumer Discretionary', 'BBY':'Consumer Discretionary', 'BN':'Consumer Discretionary',
                'CVCO':'Consumer Discretionary', 'DD':'Consumer Discretionary', 'EBAY':'Consumer Discretionary', 'PENN':'Consumer Discretionary', 'RPM':'Consumer Discretionary',
                'ZD':'Consumer Discretionary', 'ABEV':'Consumer Staples', 'BTE':'Consumer Staples', 'COKE':'Consumer Staples', 'HSY':'Consumer Staples', 'NOMD':'Consumer Staples',
                'PEP':'Consumer Staples', 'PRMW':'Consumer Staples', 'TSN':'Consumer Staples', 'CRK':'Energy', 'CTRA':'Energy', 'EOG':'Energy', 'MPC':'Energy', 'PXD':'Energy', 'SHEL':'Energy',
                'SUN':'Energy', 'UGP':'Energy', 'AGO':'Finance', 'AMG':'Finance', 'BBDO':'Finance', 'CATY':'Finance', 'CRVL':'Finance', 'EXG':'Finance', 'GBDC':'Finance', 'IBTX':'Finance',
                'UMBF':'Finance', 'ACHC':'Health Care', 'AMED':'Health Care', 'ASND':'Health Care', 'ILMN':'Health Care', 'ITGR':'Health Care', 'OMCL':'Health Care', 'PDCO':'Health Care',
                'TFX':'Health Care', 'CF':'Industrials', 'DY':'Industrials', 'HUN':'Industrials', 'HXL':'Industrials', 'THO':'Industrials', 'TRMB':'Industrials', 'UNP':'Industrials', 'WMS':'Industrials',
                'XPO':'Industrials', 'BBWI':'Misc.', 'CASY':'Misc.', 'CHPT':'Misc.', 'FCFS':'Misc.', 'LESL':'Misc.', 'NCR':'Misc.', 'ODP':'Misc.', 'WOOF':'Misc.', 'BRX':'Real Estate', 'BXP':'Real Estate',
                'EDU':'Real Estate', 'HR':'Real Estate', 'HST':'Real Estate', 'SBRA':'Real Estate', 'SLG':'Real Estate', 'UDR':'Real Estate', 'AMD':'Technology', 'CACI':'Technology', 'DELL':'Technology',
                'FIVN':'Technology', 'FTNT':'Technology', 'LPL':'Technology', 'NOW':'Technology', 'PSTG':'Technology', 'SSNC':'Technology', 'ESE':'Telecomunications', 'HPE':'Telecomunications', 'KT':'Telecomunications',
                'TIMB':'Telecomunications', 'TKC':'Telecomunications', 'VOD':'Telecomunications', 'VZ':'Telecomunications', 'WBD':'Telecomunications', 'AES':'Utilities', 'BEP':'Utilities', 'CMS':'Utilities', 'EBR':'Utilities',
                'EIX':'Utilities', 'FN':'Utilities', 'NEE':'Utilities', 'WMB':'Utilities'}

combined_df = pd.DataFrame()

for file in os.listdir(path):
    if file.endswith(".csv"):
        stock_df = pd.read_csv(os.path.join(path, file))
        stock_name = os.path.splitext(file)[0]
        stock_df["Stock"] = stock_name

        sector = sector_dict.get(stock_name)
        stock_df["Sector"] = sector

        combined_df = combined_df.append(stock_df, ignore_index=True)
        combined_df = combined_df.reset_index(drop=True)

        combined_df = combined_df.drop('Stock', axis = 1)

df = combined_df[['Date', 'Close', 'Symbol', 'Sector']]
df['Daily Return'] = df.groupby('Symbol')['Close'].pct_change()
df_grouped = df.groupby('Sector')['Daily Return'].std().reset_index()
df_grouped = df_grouped.sort_values(by='Daily Return', ascending=True)
plt.bar(df_grouped['Sector'], df_grouped['Daily Return'], align='edge')

plt.xticks(rotation=45, ha='right')
plt.title('Standard Deviation of Daily Returns by Sector')
plt.xlabel('Sector')
plt.ylabel('Standard Deviation of Daily Returns')

plt.subplots_adjust(left=0.2, right=0.9, bottom=0.3, top=0.9, wspace=0.4)

plt.show()

