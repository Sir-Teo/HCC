import pandas as pd
from datetime import datetime

# Load the Excel file
df = pd.read_excel('/gpfs/data/shenlab/wz1492/HCC/data/Recurrence_HCC_Deidentified.xlsx')  # Replace with your actual file name


def calculate_time(row):
    treatment_date = pd.to_datetime(row['DATE OF TXP'], errors='coerce')
    if pd.isna(treatment_date):
        # Handle missing treatment dates if necessary
        return None
    
    if row['recurrence post tx'] == 1:
        event_date = pd.to_datetime(row['Recurrence Date'], errors='coerce')
        if pd.isna(event_date):
            # If recurrence date is missing but event is indicated, treat as censored
            event_date = pd.to_datetime(row['Last date of followup'], errors='coerce')
            event = 0
        else:
            event = 1
    else:
        event_date = pd.to_datetime(row['Last date of followup'], errors='coerce')
        event = 0
    
    if pd.isna(event_date):
        # Handle missing event or follow-up dates
        return None
    
    # Calculate time in days; you can convert to years or months if preferred
    time = (event_date - treatment_date).days
    return time, event

# Apply the function to create 'time' and 'event' columns
df[['time', 'event']] = df.apply(
    lambda row: pd.Series(calculate_time(row)),
    axis=1
)

# Drop rows with missing time or event
df = df.dropna(subset=['time', 'event'])

# Convert 'time' to numeric (int)
df['time'] = df['time'].astype(int)

# Ensure 'event' is integer
df['event'] = df['event'].astype(int)

# Optional: Save the processed DataFrame
df.to_csv('processed_patient_labels.csv', index=False)
