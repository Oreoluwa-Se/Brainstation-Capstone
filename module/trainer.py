from .estop import EarlyStopping
from module.preprocess.load_and_batch import DataBatcher
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
from utils import config
from tqdm import tqdm
import os
import torch


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        loader: DataBatcher = None,
        estop: EarlyStopping = None,
        writer: SummaryWriter = None,
        device=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader = loader
        self.device = device
        self.writer = writer
        self.estop = estop
        self.use_auto_cast = config.MODEL_ARGS["use_amp"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_auto_cast else None

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_samples = 0
        loss_out = {"tot_loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        with tqdm(total=self.loader.train.shape[0], desc="Training") as pbar:

            total_count = self.loader.train.data.height

            while True:
                batch = self.loader.get_meta_data(
                    batch_size=config.DATASET_CONFIG["batch"],
                    data_type="train",
                    ignore_list=["case_id"],
                    output_list=["WEEK_NUM", "target"],
                )

                if total_samples >= total_count:
                    break

                # Ensure this iterates over batches
                batch_len = len(batch.texts)

                total_samples += batch_len

                self.optimizer.zero_grad()

                with autocast(enabled=self.use_auto_cast):
                    _, loss = self.model(batch)

                if self.scaler:
                    self.scaler.scale(loss["tot_loss"]).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss["tot_loss"].backward()
                    self.optimizer.step()

                for key in loss_out:
                    loss_out[key] += loss[key].item() * batch_len

                # determines if we undersample or oversample or just maintain
                self.loader.train.choose_sampling_strat(
                    precision=loss["precision"].item(),
                    recall=loss["recall"].item(),
                )

                # epoch update
                pbar.update(batch_len)

        for key in loss_out:
            loss_out[key] = loss_out[key] / total_samples

        return loss_out

    def evaluate_one_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_samples = 0
        loss_out = {"tot_loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        total_count = self.loader.valid.data.height
        with torch.no_grad():
            with tqdm(total=self.loader.valid.shape[0], desc="Validation") as pbar:
                while True:
                    batch = self.loader.get_meta_data(
                        batch_size=config.DATASET_CONFIG["batch"],
                        data_type="valid",
                        ignore_list=["case_id"],
                        output_list=["WEEK_NUM", "target"],
                    )

                    if total_samples >= total_count:
                        break

                    batch_len = len(batch.texts)
                    total_samples += batch_len

                    with autocast(enabled=self.use_auto_cast):
                        _, loss = self.model(batch)

                    for key in loss_out:
                        loss_out[key] += loss[key].item() * batch_len

                    pbar.update(batch_len)

        for key in loss_out:
            loss_out[key] = loss_out[key] / total_samples

        return loss_out

    def tensorboard_writer(self, v_info, t_info, epoch):
        # loss track
        self.writer.add_scalars(
            "Loss",
            {"train": t_info["tot_loss"], "val": v_info["tot_loss"]},
            epoch,
        )

        # accuracy track
        self.writer.add_scalars(
            "Accuracy",
            {"train": t_info["accuracy"], "val": v_info["accuracy"]},
            epoch,
        )

        tags = ["Train", "Validation"]
        for idx, c_dict in enumerate([t_info, v_info]):
            self.writer.add_scalar(f"Precision/{tags[idx]}", c_dict["precision"], epoch)
            self.writer.add_scalar(f"Recall/{tags[idx]}", c_dict["recall"], epoch)
            self.writer.add_scalar(f"F1 Score/{tags[idx]}", c_dict["f1"], epoch)

    def save_model_checkpoint(self, epoch, num_epochs, folder="model_info"):
        # Create the directory if it does not exist
        os.makedirs(folder, exist_ok=True)

        # Determine the path based on the epoch
        if epoch == 0:
            filename = "first_run.pt"
        elif epoch == num_epochs - 1:
            filename = "final_model_checkpoint.pt"
        else:
            return

        final_path = os.path.join(folder, filename)
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        print(f"Model checkpoint saved to {final_path}")

    def train(self, num_epochs: int, verbose: bool = False):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            valid_loss = self.evaluate_one_epoch()

            if verbose:
                print(
                    f"Epoch {epoch+1}, Train Loss: {train_loss['tot_loss']:.4f}, Val Loss: {valid_loss['tot_loss']:.4f}"
                )

            if self.writer:
                self.tensorboard_writer(valid_loss, train_loss, epoch)

            if self.scheduler:
                self.scheduler.step(valid_loss["tot_loss"])

            if self.estop:
                self.estop(
                    valid_loss["tot_loss"], self.model, self.optimizer, self.scheduler
                )
                if self.estop.early_stop:
                    print("Early stopping triggered")
                    break

            self.loader.reset(only_indexes=True, training=True)

            self.save_model_checkpoint(epoch, num_epochs)

        if self.writer:
            self.writer.close()

    def load_model(self, for_inference=False):
        """
        Load the model.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model (FullProcess): The model instance to load the state into.
            optimizer (Optimizer, optional): The optimizer instance.
            scheduler (lr_scheduler, optional): The scheduler instance.
            for_inference (bool): indicates if model is being loaded for inference or training.

        Returns:
            model: The model loaded with checkpoint weights.
            optimizer: The optimizer loaded with checkpoint state.
            scheduler: The scheduler loaded with checkpoint state.
        """
        checkpoint = torch.load(self.estop.checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if not for_inference:
            if self.optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
