import pandas as pd
import numpy as np
import seaborn as sns

print("Pandas version:", pd.__version__)
print("Libraries loaded successfully!")

import pandas as pd
df = pd.read_csv('african-hypertension-cvd-dataset.csv')
print("Dataset loaded! Here are the first 5 rows:")
print(df.head())
