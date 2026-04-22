# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 22.04.2026

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv("/content/amazon_sales_dataset.csv")

# Clean column names
data.columns = data.columns.str.strip()
print("Columns:", data.columns)


data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')

# Drop invalid dates
data = data.dropna(subset=['order_date'])

# Set index
data.set_index('order_date', inplace=True)

# Sort by date (important)
data = data.sort_index()

data = data[~data.index.duplicated(keep='first')]

col = 'total_revenue'

print("Using column:", col)

# Regular Differencing
data['sales_diff'] = data[col] - data[col].shift(1)

# Seasonal Decomposition
result = seasonal_decompose(data[col], model='additive', period=12)
data['sales_sea_diff'] = result.resid

# Log Transformation
data['sales_log'] = np.log(data[col])

# Log Differencing
data['sales_log_diff'] = data['sales_log'] - data['sales_log'].shift(1)

# Seasonal decomposition on log differenced
result2 = seasonal_decompose(data['sales_log_diff'].dropna(), model='additive', period=12)
data['sales_log_seasonal_diff'] = result2.resid

# Plotting
plt.figure(figsize=(16,16))

plt.subplot(6,1,1)
plt.plot(data[col])
plt.title('Original Data')

plt.subplot(6,1,2)
plt.plot(data['sales_diff'])
plt.title('Regular Differencing')

plt.subplot(6,1,3)
plt.plot(data['sales_sea_diff'])
plt.title('Seasonal Adjustment')

plt.subplot(6,1,4)
plt.plot(data['sales_log'])
plt.title('Log Transformation')

plt.subplot(6,1,5)
plt.plot(data['sales_log_diff'])
plt.title('Log + Differencing')

plt.subplot(6,1,6)
plt.plot(data['sales_log_seasonal_diff'])
plt.title('Final Stationary')

plt.tight_layout()
plt.show()
```

### OUTPUT:

REGULAR DIFFERENCING:
<img width="1389" height="253" alt="image" src="https://github.com/user-attachments/assets/a6511362-fc42-486d-86ec-e5a23a7bc6c4" />


SEASONAL ADJUSTMENT:

<img width="1379" height="248" alt="image" src="https://github.com/user-attachments/assets/c69bc6b9-23fe-4d95-9086-bbb21a09ecd8" />

LOG TRANSFORMATION:

<img width="1359" height="466" alt="image" src="https://github.com/user-attachments/assets/8cf48f1b-9dbe-4cb4-8945-ab910165e435" />


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
