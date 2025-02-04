# callbacks/custom_callbacks.py
import torch
import torchtuples as tt

DEBUG = False

class GradientClippingCallback(tt.callbacks.Callback):
    def __init__(self, clip_value):
        super().__init__()
        self.clip_value = clip_value

    def on_batch_end(self):
        torch.nn.utils.clip_grad_norm_(self.model.net.parameters(), self.clip_value)
        total_norm = 0.0
        for p in self.model.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if DEBUG:
            print(f"[DEBUG] GradientClippingCallback - Batch gradient norm: {total_norm:.4f}")

class LossLogger(tt.callbacks.Callback):
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self):
        log_df = self.model.log.to_pandas()
        if not log_df.empty:
            current_loss = log_df.iloc[-1]['train_loss']
            print(f"Epoch {len(self.epoch_losses)} - Train Loss: {current_loss:.4f}")
            self.epoch_losses.append(current_loss)
            total_norm = 0.0
            for p in self.model.net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Epoch {len(self.epoch_losses)} - Gradient Norm: {total_norm:.4f}")

            if hasattr(self.model, "x_train_std") and self.model.x_train_std is not None:
                sample_input = torch.tensor(self.model.x_train_std[:5]).to(next(self.model.net.parameters()).device)
                with torch.no_grad():
                    risk_scores = self.model.net(sample_input)
                print(f"[DEBUG] Sample risk scores: mean={risk_scores.mean().item():.4f}, "
                      f"std={risk_scores.std().item():.4f}, min={risk_scores.min().item():.4f}, "
                      f"max={risk_scores.max().item():.4f}")

class ParamCheckerCallback(tt.callbacks.Callback):
    def on_epoch_end(self):
        for name, param in self.model.net.named_parameters():
            if torch.isnan(param).any():
                print(f"[DEBUG] Parameter {name} contains NaNs.")
