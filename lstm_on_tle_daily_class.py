import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
import copy

from preprocessing_data import Preprocessor
from matplotlib import pyplot as plt

# ------------------ Dataset ------------------
class SeqDS(Dataset):
    def __init__(self, Xw, yw):
        self.Xw, self.yw = Xw, yw

    def __len__(self):
        return len(self.Xw)

    def __getitem__(self, i):
        return torch.from_numpy(self.Xw[i]), torch.from_numpy(self.yw[i])


# ------------------ Model ------------------
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.drop(h[-1])
        return self.fc(h)  # logits


# ------------------ Pipeline Class ------------------
class LSTMTLEDailyPipeline:
    def __init__(
        self,
        unprop_csv_path="satellite_data/orbital_elements/unpropagated_elements_CryoSat-2.csv",
        man_path="satellite_data/manoeuvres/cs2man.txt",
        time_col=None,
        label_col="maneuver_status",
        base_feats=None,
        add_diffs=True,
        only_diffs = False,
        expand_maneuver_days=0,
        seq_len=64,
        stride=1,
        batch_size=128,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        hidden=64,
        layers=2,
        dropout=0.2,
        val_frac=0.20,
        test_frac=0.20,
        seed=42,
        patience=6,
        device=None
    ):
        """
        LSTM pipeline for daily TLE data with maneuver detection.
        Parameters:
        ----------
        unprop_csv_path: str
            Path to the CSV file containing unpropagated orbital elements.
        man_path: str
            Path to the file containing maneuver data.
        time_col: str or None
            Name of the time column to set as index. If None, assumes index is already datetime
        label_col: str
            Name of the label column indicating maneuver status.
        base_feats: list of str or None
            List of base feature column names to use. If None, defaults to a predefined set.
        add_diffs: bool
            Whether to add first differences of base features as additional channels.
        seq_len: int
            Length of the input sequences for the LSTM.
        stride: int
            Stride for the sliding window when creating sequences.
        batch_size: int
            Batch size for training.
        epochs: int
            Number of training epochs.
        lr: float
            Learning rate for the optimizer.
        weight_decay: float
            Weight decay (L2 regularization) for the optimizer.
        hidden: int
            Number of hidden units in the LSTM.
        layers: int
            Number of LSTM layers.
        dropout: float
            Dropout rate for the LSTM and fully connected layers.
        val_frac: float
            Fraction of the training data to use for validation.
        test_frac: float
            Fraction of the data to use for testing.
        seed: int
            Random seed for reproducibility.
        patience: int
            Number of epochs to wait for improvement before early stopping.
        device: str or None
            Device to run the model on ("cuda" or "cpu"). If None, automatically selects.
        only_diffs: bool
            Whether to use only the first differences of base features as input.
        """
        # Paths & columns
        self.unprop_csv_path = unprop_csv_path
        self.man_path = man_path
        self.time_col = time_col
        self.label_col = label_col

        if base_feats is None:
            base_feats = [
                "inclination",
                "Brouwer mean motion",
                "specific_angular_momentum",
            ]
        self.base_feats = base_feats
        self.add_diffs = add_diffs

        # Training hyperparams
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.patience = patience

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Will be filled during run
        self.scaler = None
        self.model = None
        self.train_threshold = None
        # Set seeds
        self._set_seeds()
        self.only_diffs = only_diffs
        self.expand_maneuver_days = expand_maneuver_days
    # ------------------ Utils ------------------
    def _set_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def time_split(df: pd.DataFrame, test_frac=0.2):
        df = df.sort_index()
        cut = int(len(df) * (1 - test_frac))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    @staticmethod
    def make_windows(X: np.ndarray, y: np.ndarray, seq_len=32, stride=1):
        Xw, yw = [], []
        for s in range(0, len(X) - seq_len + 1, stride):
            e = s + seq_len
            Xw.append(X[s:e])
            yw.append(y[e - 1])  # label at window end
        return np.asarray(Xw, np.float32), np.asarray(yw, np.float32).reshape(-1, 1)

    @staticmethod
    def pick_threshold(scores, y_true):
        prec, rec, thr = precision_recall_curve(y_true, scores)
        f1 = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
        thr = thr[np.argmax(f1)] if len(thr) else float(np.median(scores))
        return thr


    def eval_block(self, scores: np.ndarray, y_true: np.ndarray, desc: str, thr=None):
        """
        Evaluate scores against true labels, optionally using a provided threshold.
        If no threshold is provided, pick one based on training data."""

        if thr is None:
            thr = self.pick_threshold(scores, y_true)
            self.train_threshold = thr
        yhat = (scores >= thr).astype(int)
        auprc = average_precision_score(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        f1 = f1_score(y_true, yhat)
        sens = np.sum((yhat == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))
        spec = np.sum((yhat == 0) & (y_true == 0)) / max(1, np.sum(y_true == 0))

        print(
            f"[{desc}] thr={thr:.4f}  AUPRC={auprc:.4f}  "
            f"ROC AUC={roc_auc:.4f}  F1={f1:.4f}  "
            f"Sensitivity={sens:.4f}  Specificity={spec:.4f}"
        )
        return {
            "thr": thr,
            "AUPRC": auprc,
            "ROCAUC": roc_auc,
            "F1": f1,
            "Sensitivity": sens,
            "Specificity": spec,
        }
    
    @staticmethod
    def soft_match_scores(score, y_true, window=3, thr=None):
        """
        Compute soft-matching TP, FP, FN, precision, recall, F1 for two binary sequences.
        
        - y_true: ground truth binary array.
        - score: predicted scores (float array); thresholded at 0.5 to get binary predictions.
        - window: integer tolerance for matching (default ±3 indices).
        """
        if thr is None:
            thr = LSTMTLEDailyPipeline.pick_threshold(score, y_true)
        y_pred = (score >= thr).astype(int)
        if len(y_pred) != len(y_true):
            raise ValueError("y_true and y_pred must have the same length.")
        # Find all indices of positive events in true and predicted arrays
        pred_indices = [i for i, v in enumerate(y_pred) if v == 1]
        true_indices = [j for j, v in enumerate(y_true) if v == 1]

        matched_pred = set()
        matched_true = set()
        candidate_pairs = []
        # Build list of (distance, pred_idx, true_idx) for all pairs within the window
        for i in pred_indices:
            for j in true_indices:
                dist = abs(i - j)
                if dist <= window:
                    candidate_pairs.append((dist, i, j))
        # Sort by temporal distance (closest matches first)
        candidate_pairs.sort(key=lambda x: x[0])

        # Greedily match pairs (smallest distance first), without reuse of indices
        tp = 0
        for dist, i, j in candidate_pairs:
            if i not in matched_pred and j not in matched_true:
                matched_pred.add(i)
                matched_true.add(j)
                tp += 1

        total_preds = len(pred_indices)
        total_true = len(true_indices)
        fp = total_preds - tp
        fn = total_true - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        print (f"Soft-match results (window={window}): Threshold {thr:.4f}, TP={tp}, FP={fp}, FN={fn}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return tp, fp, fn, precision, recall, f1

    def train_epoch(self, model, loader, opt, crit):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logit = model(xb)
            loss = crit(logit, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item() * len(xb)
        return tot / len(loader.dataset)

    @torch.no_grad()
    def scores_on(self, model, loader):
        model.eval()
        s, y = [], []
        for xb, yb in loader:
            p = torch.sigmoid(model(xb.to(self.device))).cpu().numpy().reshape(-1)
            s.append(p)
            y.append(yb.numpy().reshape(-1))
        return np.concatenate(s), np.concatenate(y)

    # ------------------ Data Loading ------------------
    def load_and_preprocess(self):
        # Load
        cs2_prep = Preprocessor(self.unprop_csv_path, self.man_path)
        cs2_prep.processed_data()
        df = cs2_prep.select_features( self.base_feats )
        # Handle time index if needed
        if self.time_col is not None:
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df = df.set_index(self.time_col)

        df = df.sort_index()

        # Sanity check
        assert all(col in df.columns for col in self.base_feats + [self.label_col])

        # Optionally add first differences
        feats = self.base_feats.copy()
        if self.add_diffs:
            for c in self.base_feats:
                dcol = f"Δ {c}"
                df[dcol] = df[c].astype(float).diff().fillna(0.0)
                feats.append(dcol)
        if self.only_diffs:
            diff_feats = [f"Δ {c}" for c in self.base_feats]
            df = df[diff_feats + [self.label_col]]
            feats = diff_feats
        return df, feats
    
    @staticmethod
    def expand_labels_forward(maneuvers_label, expand: int = 2) -> np.ndarray:
        """
        Expand each positive label forward by `expand` steps only.

        Args:
            y (np.ndarray): 1D array of 0/1 labels.
            expand (int): Number of future days to mark as 1 after a positive label.

        Returns:
            np.ndarray: New array with forward-expanded labels.
        """
        y = np.asarray(maneuvers_label, dtype=int)
        y_new = y.copy()
        n = len(y)
        for t in np.where(y == 1)[0]:
            end = min(t + expand + 1, n)
            y_new[t:end] = 1
        return y_new

    def split_and_scale(self, df, feats):
        # Time-based split: train_full vs test
        df_tr_full, df_te = self.time_split(df, self.test_frac)
        # From train_full, carve a validation tail
        cut = int(len(df_tr_full) * (1 - self.val_frac))
        df_tr = df_tr_full.iloc[:cut].copy()
        df_va = df_tr_full.iloc[cut:].copy()


        
        # Train-only scaling
        self.scaler = StandardScaler().fit(df_tr[feats].values.astype(float))
        X_tr = self.scaler.transform(df_tr[feats].values.astype(float))
        X_va = self.scaler.transform(df_va[feats].values.astype(float))
        X_te = self.scaler.transform(df_te[feats].values.astype(float))

        y_tr = df_tr[self.label_col].values.astype(int)
        # expand train labels forward if needed
        if self.expand_maneuver_days > 0:
            y_tr = self.expand_labels_forward(y_tr, expand=self.expand_maneuver_days)

        y_va = df_va[self.label_col].values.astype(int)
        y_te = df_te[self.label_col].values.astype(int)

        # Windows
        Xw_tr, yw_tr = self.make_windows(
            X_tr, y_tr, self.seq_len, self.stride
        )
        Xw_va, yw_va = self.make_windows(
            X_va, y_va, self.seq_len, self.stride
        )
        Xw_te, yw_te = self.make_windows(
            X_te, y_te, self.seq_len, self.stride
        )

        return (Xw_tr, yw_tr), (Xw_va, yw_va), (Xw_te, yw_te)

    # ------------------ Train / Evaluate ------------------
    def build_model_and_loss(self, in_dim, yw_tr):
        # Class weight
        n_pos = int(yw_tr.sum())
        n_neg = int(len(yw_tr) - n_pos)
        pos_w = max(1.0, n_neg / max(1, n_pos))

        model = LSTMClassifier(
            in_dim=in_dim,
            hidden=self.hidden,
            layers=self.layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_w], device=self.device)
        )

        optim = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return model, criterion, optim, pos_w

    def fit(self, tr_loader, va_loader, criterion, optim, pos_w, yw_tr, verbose=True):
        best, wait = -1.0, 0
        best_state = None

        for ep in range(1, self.epochs + 1):
            tr_loss = self.train_epoch(self.model, tr_loader, optim, criterion)
            s_va, y_va_w = self.scores_on(self.model, va_loader)
            auprc_va = average_precision_score(y_va_w, s_va)
            if verbose:
                print(
                    f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | "
                    f"val_AUPRC={auprc_va:.4f} | pos_w={pos_w:.1f} | "
                    f"positive labels={int(yw_tr.sum())}/{len(yw_tr)}"
                )

            if auprc_va > best + 1e-4:
                best, best_state, wait = auprc_va, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print("Early stopping.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def evaluate(self, loader, set_name="Validation", thr=None):
        # Validation
        s, y_w = self.scores_on(self.model, loader)
        val_metrics = self.eval_block(s, y_w, set_name, thr)
        self.soft_match_scores(s, y_w, window=3, thr=thr)
        self.plot_pr_curve(s, y_w, desc=set_name)
        return val_metrics
    
    
    @staticmethod
    def plot_pr_curve(scores, y_true, desc="Validation"):
        """
        Plot Precision–Recall curve for given scores and labels.

        Parameters
        ----------
        scores : np.ndarray
            Predicted probabilities (output of sigmoid).
        y_true : np.ndarray
            True binary labels (0/1).
        desc : str
            Description to show in the plot title (e.g. 'Validation', 'Test').
        """
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)

        plt.figure()
        # step-style PR curve
        plt.step(rec, prec, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{desc} PR curve (AP = {ap:.3f})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # ------------------ High-level run ------------------
    def run(self, verbose=True):
        # 1. Load & preprocess
        df, feats = self.load_and_preprocess()
        print(f"Using features: {feats}")
        # 2. Split & scale & window
        (Xw_tr, yw_tr), (Xw_va, yw_va), (Xw_te, yw_te) = self.split_and_scale(
            df, feats
        )

        # 3. DataLoaders
        tr_loader = DataLoader(
            SeqDS(Xw_tr, yw_tr), batch_size=self.batch_size, shuffle=True
        )
        va_loader = DataLoader(
            SeqDS(Xw_va, yw_va), batch_size=self.batch_size, shuffle=False
        )
        te_loader = DataLoader(
            SeqDS(Xw_te, yw_te), batch_size=self.batch_size, shuffle=False
        )

        # 4. Build model & loss
        in_dim = len(feats)
        self.model, criterion, optim, pos_w = self.build_model_and_loss(
            in_dim, yw_tr
        )

        # 5. Train
        self.fit(tr_loader, va_loader, criterion, optim, pos_w, yw_tr, verbose=verbose)
        
        # 6. Evaluate
        val_metrics = self.evaluate(va_loader, set_name="Validation")
        
        # 7. Combine val and train set for obtain test threshold
        Xw_trval = np.concatenate([Xw_tr, Xw_va], axis=0)
        yw_trval = np.concatenate([yw_tr, yw_va], axis=0)   
        trval_loader = DataLoader(
            SeqDS(Xw_trval, yw_trval), batch_size=self.batch_size, shuffle=False
        )
        self.evaluate(trval_loader, set_name="Train+Val")
        test_metrics = self.evaluate(te_loader, set_name="Test", thr=self.train_threshold)
        
        return test_metrics
    


if __name__ == "__main__":
    pipeline = LSTMTLEDailyPipeline(only_diffs=True, expand_maneuver_days=2)
    pipeline.run(verbose=True)
