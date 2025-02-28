import pandas as pd

# Load the Excel file
file_path = "spreadsheets/TCGA-CDR-SupplementalTableS1.xlsx"
xls = pd.ExcelFile(file_path)

# Read the sheet named "ExtraEndpoints"
df = xls.parse("ExtraEndpoints")

# Convert all 2s to 0s in column 'DFI.cr'
df['DFI.cr'] = df['DFI.cr'].replace(2, 0)

# Remove rows with NAs in 'DFI.cr' or 'DFI.time.cr'
df_cleaned = df.dropna(subset=['DFI.cr', 'DFI.time.cr'])

# Filter the dataframe to keep only rows where 'type' is 'LIHC'
df_filtered = df_cleaned[df_cleaned['type'] == 'LIHC']

# save file
df_filtered.to_csv("spreadsheets/tcga.csv", index=False)