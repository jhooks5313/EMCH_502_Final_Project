import numpy as np
from datetime import datetime
import sys
import pandas as pd

from helpers import Dataset, Logger
from models import ML_Models, ANN_model

# which problems to run
RUN_P1 = False
RUN_P2 = True

save_alarms = True     # whether to save ANN predicted alarm columns to file
plot_loss = True    # whether to plot figures
save_metrics = True     # whether to save metrics

# logging
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = f"outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
tee = Logger(log_path)
sys.stdout = tee

# load dataset
print("Loading data")
ds = Dataset()
ds.summary()

# P1 - ML models
if RUN_P1:
    print('\n' + '-'*50)
    print('EMCH 502 - Problem 1: ML Models')
    print(">")

    ml  = ML_Models(ds)
    ml.build_all()

    print('\n--- Evaluate ---')
    ml.evaluate()
    ml.feature_importance('Random Forest')
    pred_pcts = ml.predict_files()

    if plot_loss:
        ml.plot_confusion_matrices(run_id)
        ml.plot_roc_curves(run_id)
        ml.plot_feature_importance(run_id)

    if save_metrics:
        ml.save_metrics_csv(run_id, pred_pcts)

# P2 - ANN model
if RUN_P2:
    print('\n' + '-'*50)
    print('EMCH 502 - Problem 2 (Graduate): ANN')
    print(">")

    ann = ANN_model(ds, epochs=100, lr=1e-3, batch_size=64)
    ann.build_and_train()
    ann.evaluate()
    if plot_loss:
        ann.plot_loss(run_id)
        ann.plot_confusion_matrix(run_id)
        ann.plot_roc_curve(run_id)
    
    if save_metrics:
        ann.save_metrics_csv(run_id)

    if save_alarms:
        preds_06 = ann.predict_column(ds.X_pred_06)
        preds_09 = ann.predict_column(ds.X_pred_09)
        pred_df = pd.DataFrame({
            'file': ['FOC_20190801_06'] * len(preds_06) + ['FOC_20190812_09'] * len(preds_09),
            'row':  list(range(len(preds_06))) + list(range(len(preds_09))),
            'predicted_alarm': list(preds_06) + list(preds_09),
        })
        pred_df['predicted_alarm_label'] = pred_df['predicted_alarm'].map({0: 'Normal', 1: 'Fault'})
        pred_df.to_csv(f'outputs/ann_predicted_alarms_{run_id}.csv', index=False)
        print(f"Predicted alarm columns saved to outputs/ann_predicted_alarms_{run_id}.csv")

    ann.predict_files()

sys.stdout = tee.terminal
tee.close()
print(f"Log saved to {log_path}")