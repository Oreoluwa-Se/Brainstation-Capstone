import torch
import os


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        mode: str = "min",
        verbose: bool = False,
        checkpoint_path: str = "checkpth.pt",
    ):
        """
        Args:
            patience (int): Number of epochs to wait after min has been hit.
                            After this number of epochs, training stops.
            min_delta (float): Minimum change in the monitored quantity
                               to qualify as an improvement.
            mode (str): One of ['min', 'max'].
                In 'min' mode, training will stop when the quantity monitored stops decreasing.
                In 'max' mode it will stop when the quantity monitored stops increasing.
            verbose (bool): If True, prints a message for each validation loss improvement.
            checkpoint_path (str): Path to save the checkpoint model.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counter = 10
        self.early_stop = False
        self.best_score = None
        self.dir_checked = False

        if mode not in ["min", "max"]:
            raise ValueError("Mode must be min or max")

    def save_checkpoint(
        self, val_loss, model, optimizer=None, scheduler=None, loc: str = None
    ):
        """Saves model when required."""
        if self.verbose:
            print(
                f"Validation loss decreased to {val_loss}. Saving model to {self.checkpoint_path}..."
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "val_loss": val_loss,
        }

        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        store_loc = loc if loc is not None else self.checkpoint_path
        if not self.dir_checked:
            directory = os.path.dirname(store_loc)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.dir_checked = True

        torch.save(checkpoint, store_loc)

    def __call__(self, val_loss, model, optimizer=None, scheduler=None):
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None or score > self.best_score + self.min_delta:
            if self.best_score is None:
                self.best_score = score
            elif score > self.best_score + self.min_delta:
                self.best_score = score
                self.save_checkpoint(val_loss, model, optimizer, scheduler)
                self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"Early Stopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
