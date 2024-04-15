from sklearn.metrics import accuracy_score
from torch import nn, Tensor
from typing import Dict
import torch
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    """Used to calculate loss from training.
    - Uses Class Weights for handling imbalance for handling class imbalance
    - Alpha parameter is adapted based on the batch distribution
    """

    def __init__(
        self,
        reduction="mean",
        eps=1e-5,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.reduction = reduction
        self.eps = eps
        self.threshold = threshold

    def forward(self, logits: Tensor, target: Tensor):
        target = target.view(-1, 1)
        logits = logits.view(-1, 1)

        probs = torch.sigmoid(logits)

        unique_classes, counts = torch.unique(target, return_counts=True)
        total_samples = target.numel()
        class_weights = {
            label.item(): total_samples / (counts[i] * len(unique_classes))
            for i, label in enumerate(unique_classes)
        }

        # calculate a difficulty metric for each
        hardness = 1 - torch.abs(probs - self.threshold)

        weights = torch.zeros_like(probs)
        for label in unique_classes:
            label_mask = (target == label).float()
            # Incorporate class-specific weight moderated by sample-specific hardness
            weights += label_mask * (class_weights[label.item()] * hardness)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, weight=weights.detach()
        )

        if self.reduction == "mean":
            return bce_loss.mean()
        elif self.reduction == "sum":
            return bce_loss.sum()

        return bce_loss


class Loss(nn.Module):

    def __init__(
        self,
        temporal_weight: float = 1.0,
        reduction="mean",
        threshold=0.5,
    ):
        super().__init__()

        assert reduction.lower() in [
            "sum",
            "mean",
        ], "Incorrect reduction type must be [sum, mean]"

        self.bce = AdaptiveLoss(
            reduction=reduction,
            threshold=threshold,
        )

        self.reduction = reduction
        self.threshold = threshold
        self.eps = 1e-5

    def soft_f1_score(self, logits_proba: Tensor, target: Tensor):

        tp = torch.sum(logits_proba * target)
        fp = torch.sum(logits_proba * (1 - target))
        fn = torch.sum((1 - logits_proba) * target)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)

        return precision, recall, f1

    def compute_accuracy(self, proba: Tensor, targets: Tensor):
        preds = (proba > self.threshold).float()
        accuracy = torch.mean((preds == targets).float())
        return accuracy

    def forward(
        self, logits: Tensor, target: Tensor, verbose=False
    ) -> Dict[str, float]:

        # ensure same type
        target = target.type_as(logits)
        # Calculate BCE loss
        bce_loss = self.bce(logits, target[:, 1])

        # checking other parameters
        proba = torch.sigmoid(logits)
        accuracy = self.compute_accuracy(proba, target[:, 1])
        precision, recall, f1 = self.soft_f1_score(proba, target[:, 1])

        loss_out = {
            "tot_loss": bce_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        print(loss_out)
        return loss_out


def bce_only_loss():
    loss_desc = AdaptiveLoss(reduction="None")
    logits = torch.randn(10, requires_grad=True)
    target = torch.empty(10).random_(2)
    loss = loss_desc(logits, target)
    print(f"Predicted_logits:\n{logits}")
    print(f"target:\n{target}")
    print(loss)


def combined_loss():
    n = 100
    preds = torch.randn(n, 1)  # Simulated logits
    weeks = torch.randint(1, 53, (n, 1))  # Week numbers
    binary_values = torch.randint(0, 2, (n, 1))  # Binary targets
    target = torch.cat((weeks, binary_values.type_as(preds)), dim=1)

    print(f"Prediction: \n{preds}")
    print(f"Target: \n{target}")
    t_loss = Loss(temporal_weight=1.0, reduction="mean")

    out = t_loss(preds, target, verbose=True)
    print(out["tot_loss"])


if __name__ == "__main__":
    seed_number = 42
    torch.manual_seed(seed_number)

    combined_loss()
