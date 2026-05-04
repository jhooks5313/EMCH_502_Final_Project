import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from helpers import Dataset

#--------------------------------------------------------------
# P1 ML models - Logistic regression, RF, GB

class ML_Models:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.models = {}

    #----------------------------
    # build models
    def build_log_regression(self):
        model = Pipeline([
            ('scalar', StandardScaler()),
            ('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=54))
        ])
        model.fit(self.ds.X_train,self.ds.y_train)
        self.models['Logistic Regression'] = model
        return model
    
    def build_RF(self):
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=54, n_jobs=-1)
        model.fit(self.ds.X_train,self.ds.y_train)
        self.models['Random Forest'] = model
        return model
    
    def build_GB(self):
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,random_state=54)
        model.fit(self.ds.X_train,self.ds.y_train)
        self.models['Gradient Boosting'] = model
        return model
    
    def build_all(self):
        print('Training Logistic Regression')
        self.build_log_regression()
        print('Training Random Forest')
        self.build_RF()
        print('Training Gradient Boosting')
        self.build_GB()

    #----------------------------
    # evaluate
    def evaluate(self, name=None):
        targets = {name: self.models[name]} if name else self.models
        for mname, model in targets.items():
            y_pred = model.predict(self.ds.X_test)
            acc = accuracy_score(self.ds.y_test, y_pred)
            print(f"  {mname}  |  Accuracy: {acc:.4f}")
            print(classification_report(
                self.ds.y_test, y_pred, target_names=['Normal', 'Fault']
            ))
        
    def feature_importance(self,model_name='Random Forest',top_n=10):
        model = self.models[model_name]
        imp = pd.Series(model.feature_importances_,index=self.ds.feature_names)
        print(f"\nTop {top_n} features ({model_name}):")
        print(imp.nlargest(top_n).to_string())
    
    def predict_files(self):
        print("\n---EMCH 502 P1 Predictions---")
        print("Abnormal = >0.01% of rows predicted as fault\n")
        pred_pcts = {}
        for mname, model in self.models.items():
            print(f"[{mname}]")
            pct_06, v_06 = self._predict_one("FOC_20190801_06", model, self.ds.X_pred_06)
            pct_09, v_09 = self._predict_one("FOC_20190812_09", model, self.ds.X_pred_09)
            pred_pcts[mname] = (pct_06, v_06, pct_09, v_09)
        return pred_pcts
        
    @staticmethod
    def _predict_one(fname, model, X):
        fault_pct = model.predict(X).mean() * 100
        verdict = 'Yes' if fault_pct > 0.01 else 'No'
        print(f"  {fname}: {fault_pct:.1f}% fault rows → Abnormal: {verdict}")
        return fault_pct, verdict

    #----------------------------
    # plotting and metrics
    def plot_confusion_matrices(self, run_id):
        fig, axes = plt.subplots(1, len(self.models), figsize=(5 * len(self.models), 4))
        if len(self.models) == 1:
            axes = [axes]
        for ax, (mname, model) in zip(axes, self.models.items()):
            y_pred = model.predict(self.ds.X_test)
            cm = confusion_matrix(self.ds.y_test, y_pred)
            ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Fault']).plot(ax=ax, colorbar=False)
            ax.set_title(mname)
        plt.suptitle('Confusion Matrices — P1 ML Models')
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/p1_confusion_{run_id}.png', dpi=150)
        plt.show()

    def plot_roc_curves(self, run_id):
        plt.figure(figsize=(7, 5))
        for mname, model in self.models.items():
            # use predict_proba if available, else decision_function
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(self.ds.X_test)[:, 1]
            else:
                scores = model.decision_function(self.ds.X_test)
            fpr, tpr, _ = roc_curve(self.ds.y_test, scores)
            plt.plot(fpr, tpr, label=f'{mname} (AUC={auc(fpr, tpr):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves — P1 ML Models')
        plt.legend()
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/p1_roc_{run_id}.png', dpi=150)
        plt.show()

    def plot_feature_importance(self, run_id, model_name='Random Forest', top_n=15):
        model = self.models[model_name]
        imp = pd.Series(model.feature_importances_, index=self.ds.feature_names).nlargest(top_n)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp[::-1].plot(kind='barh', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances ({model_name})')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/p1_feature_importance_{run_id}.png', dpi=150)
        plt.show()

    def save_metrics_csv(self, run_id, pred_pcts):
        rows = []
        for mname, model in self.models.items():
            y_pred = model.predict(self.ds.X_test)
            p, r, f, _ = precision_recall_fscore_support(
                self.ds.y_test, y_pred, labels=[1], zero_division=0
            )
            acc = accuracy_score(self.ds.y_test, y_pred)
            pct_06, verdict_06, pct_09, verdict_09 = pred_pcts[mname]
            rows.append({
                'run_id': run_id, 'model': mname,
                'accuracy': round(acc, 4),
                'precision_fault': round(p[0], 4),
                'recall_fault': round(r[0], 4),
                'f1_fault': round(f[0], 4),
                'pred_06_pct': round(pct_06, 2), 'verdict_06': verdict_06,
                'pred_09_pct': round(pct_09, 2), 'verdict_09': verdict_09,
            })
        metrics_path = 'outputs/p1_metrics.csv'
        os.makedirs('outputs', exist_ok=True)
        df = pd.DataFrame(rows)
        write_header = not os.path.exists(metrics_path)
        df.to_csv(metrics_path, mode='a', header=write_header, index=False)
        print(f"P1 metrics saved to {metrics_path}")

#--------------------------------------------------------------
# P2 ANN model - Logistic regression, RF, GB

class _FaultNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
    
class ANN_model:
    def __init__(self, dataset:Dataset, epochs=100, lr=1e-3, batch_size=64):
        self.ds = dataset
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.train_losses = []
        self.val_losses   = []
    
    def _to_tensors(self, X, y=None):
        X_s = self. scaler.transform(X)
        X_t = torch.tensor(X_s, dtype=torch.float32).to(self.device)
        if y is not None:
            y_t = torch.tensor(y.values, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t
    
    def build_and_train(self):
        self.scaler.fit(self.ds.X_train)
        X_tr, y_tr = self._to_tensors(self.ds.X_train, self.ds.y_train)
        X_val, y_val = self._to_tensors(self.ds.X_test, self.ds.y_test)

        n_fault = (self.ds.y_train == 1).sum()
        n_normal = (self.ds.y_train == 0).sum()
        pos_weight = torch.tensor([n_normal/max(n_fault,1)], dtype=torch.float32).to(self.device)

        self.model = _FaultNet(X_tr.shape[1]).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loader     = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)

        print(f"\nTraining ANN on {self.device} for {self.epochs} epochs")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val), y_val).item()

            self.train_losses.append(epoch_loss / len(X_tr))
            self.val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:>3}/{self.epochs} — "
                      f"train: {self.train_losses[-1]:.4f}  val: {val_loss:.4f}")
        
    def evaluate(self):
        self.model.eval()
        X_val, _ = self._to_tensors(self.ds.X_test, self.ds.y_test)
        with torch.no_grad():
            preds = (torch.sigmoid(self.model(X_val)) >= 0.5).cpu().numpy().astype(int)
        acc = accuracy_score(self.ds.y_test, preds)
        print(f"\n  ANN  |  Accuracy: {acc:.4f}")
        print(classification_report(self.ds.y_test, preds, target_names=['Normal', 'Fault']))

    def predict_files(self):
        self.model.eval()
        print("\n--- ANN P1 Predictions ---")
        for fname, X in [("FOC_20190801_06", self.ds.X_pred_06),
                         ("FOC_20190812_09", self.ds.X_pred_09)]:
            Xt = self._to_tensors(X)
            with torch.no_grad():
                fault_pct = (torch.sigmoid(self.model(Xt)) >= 0.5).float().mean().item() * 100
            verdict = 'Yes' if fault_pct > 1 else 'No'
            print(f"  {fname}: {fault_pct:.1f}% fault rows → Abnormal: {verdict}")

    def predict_column(self,X):
        self.model.eval()
        Xt = self._to_tensors(X)
        with torch.no_grad():
            preds = (torch.sigmoid(self.model(Xt)) >= 0.5).cpu().numpy().astype(int)
        return preds
    
    #----------------------------
    # plotting and metrics
    def plot_loss(self, run_id=''):
        plt.figure(figsize=(8,4))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses,   label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')
        plt.title('ANN Training & Validation Loss')
        plt.legend()
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/ann_loss_{run_id}.png', dpi=150)
        plt.show()

    def plot_confusion_matrix(self, run_id):
        self.model.eval()
        X_val, _ = self._to_tensors(self.ds.X_test, self.ds.y_test)
        with torch.no_grad():
            preds = (torch.sigmoid(self.model(X_val)) >= 0.5).cpu().numpy().astype(int)
        cm = confusion_matrix(self.ds.y_test, preds)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Fault']).plot(ax=ax, colorbar=False)
        ax.set_title('Confusion Matrix — ANN')
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/ann_confusion_{run_id}.png', dpi=150)
        plt.show()

    def plot_roc_curve(self, run_id):
        self.model.eval()
        X_val, _ = self._to_tensors(self.ds.X_test, self.ds.y_test)
        with torch.no_grad():
            scores = torch.sigmoid(self.model(X_val)).cpu().numpy()
        fpr, tpr, _ = roc_curve(self.ds.y_test, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ANN (AUC={auc(fpr, tpr):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve — ANN')
        plt.legend()
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/ann_roc_{run_id}.png', dpi=150)
        plt.show()

    def save_metrics_csv(self, run_id):
        self.model.eval()
        X_val, _ = self._to_tensors(self.ds.X_test, self.ds.y_test)
        with torch.no_grad():
            preds = (torch.sigmoid(self.model(X_val)) >= 0.5).cpu().numpy().astype(int)
        acc = accuracy_score(self.ds.y_test, preds)
        p, r, f, _ = precision_recall_fscore_support(
            self.ds.y_test, preds, labels=[1], zero_division=0
        )
        # file verdicts
        results = {}
        for fname, X in [('06', self.ds.X_pred_06), ('09', self.ds.X_pred_09)]:
            Xt = self._to_tensors(X)
            with torch.no_grad():
                pct = (torch.sigmoid(self.model(Xt)) >= 0.5).float().mean().item() * 100
            results[fname] = (round(pct, 2), 'Yes' if pct > 1 else 'No')

        row = {
            'run_id': run_id, 'epochs': self.epochs, 'lr': self.lr,
            'batch_size': self.batch_size,
            'accuracy': round(acc, 4),
            'precision_fault': round(p[0], 4),
            'recall_fault': round(r[0], 4),
            'f1_fault': round(f[0], 4),
            'final_train_loss': round(self.train_losses[-1], 4),
            'final_val_loss': round(self.val_losses[-1], 4),
            'pred_06_pct': results['06'][0], 'verdict_06': results['06'][1],
            'pred_09_pct': results['09'][0], 'verdict_09': results['09'][1],
        }
        metrics_path = 'outputs/ann_metrics.csv'
        os.makedirs('outputs', exist_ok=True)
        df = pd.DataFrame([row])
        write_header = not os.path.exists(metrics_path)
        df.to_csv(metrics_path, mode='a', header=write_header, index=False)
        print(f"ANN metrics saved to {metrics_path}")

