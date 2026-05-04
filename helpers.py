import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Logger:
    # Writes output to both terminal and a log file 
    # Ref: https://github.com/efratoio/NLPProj/blob/5beddec5824d543da3399f3ea861f76aa60e5bb6/finalproj/main.py
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

class Dataset:
    ALWAYS_DROP = [
        'DATE',
        'FRONT IN', 'REAR_IN', 'MACHINE TEMP', 'VIBRATION',   # 100% missing
        'CUTTING REMOVAL AMOUNT', 'MSPINDLE LOAD',             # 100% missing
        'MAIN COMMENT', 'EXEC BLOCK',                          # free-text
        'MACHINE ALARM', 'MACHINE WARNING',                    # target
        'MAIN PROGRAM', 'EXEC PROGRAM', 'SEQ NUMBER',          # program IDs
    ]
    CATEGORICAL_COLS = ['SPINDLE DIRECTION', 'NC MODE', 'NC STATUS', 'FEED AXIS STATUS']

    def __init__(self, data_dir='data', test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.X_pred_06 = None
        self.X_pred_09 = None
        self.feature_names = None

        self._load_and_prepare()
    
    def _load_csv(self, name):
        return pd.read_csv(os.path.join(self.data_dir, name), low_memory=False)

    def _make_label(self, df):
        return df['MACHINE ALARM'].astype(str).str.strip().isin(['70008', '070008']).astype(int)

    def _encode_categoricals(self, df):
        df = df.copy()
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).str.strip())
        return df

    def _clean(self, df):
        cols_to_drop = [c for c in self.ALWAYS_DROP if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        df = self._encode_categoricals(df)
        df = df.fillna(df.median(numeric_only=True))
        return df
    
    def _load_and_prepare(self):
        df15 = self._load_csv('FOC_20190801_15.csv')
        df16 = self._load_csv('FOC_20190801_16.csv')
        df06 = self._load_csv('FOC_20190801_06.csv')
        df09 = self._load_csv('FOC_20190812_09.csv')

        y15 = self._make_label(df15)
        y16 = self._make_label(df16)

        df15_c = self._clean(df15)
        df16_c = self._clean(df16)
        df06_c = self._clean(df06)
        df09_c = self._clean(df09)

        X_all  = pd.concat([df15_c, df16_c], ignore_index=True)
        y_all  = pd.concat([y15,    y16],    ignore_index=True)

        self.feature_names = list(X_all.columns)
        # Align prediction frames to training columns
        for pred_df in [df06_c, df09_c]:
            for col in self.feature_names:
                if col not in pred_df.columns:
                    pred_df[col] = 0

        self.X_pred_06 = df06_c[self.feature_names]
        self.X_pred_09 = df09_c[self.feature_names]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_all, y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all,
        )
    
    def summary(self):
        print(f"Train: {len(self.X_train)} rows | Test: {len(self.X_test)} rows")
        print(f"Fault rate (train): {self.y_train.mean()*100:.1f}%")
        print(f"Features: {len(self.feature_names)}")

