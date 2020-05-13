import pandas as pd

fld = 'hum_train.csv'
df = pd.read_csv(fld)

df['RISK_MM'] = df['TOTAL_PRECIPITATION'].shift(-1)


# other 2 columns
#1- pr today
def f(row):
    if row['TOTAL_PRECIPITATION'] >=  1.1:
        val = 1
    else:
        val = 0
    return val

df['PR_TODAY'] = df.apply(f, axis=1)

#2- pr tmmrw
def f(row):
    if row['RISK_MM'] >=  1.1:
        val = 1
    else:
        val = 0
    return val

df['PR_TMMRW'] = df.apply(f, axis=1)

df.to_csv('hum_train_final.csv')

