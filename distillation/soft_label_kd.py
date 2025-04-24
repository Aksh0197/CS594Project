import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLabelDistillationLoss(nn.Module):
    def _init_(self, temperature=4.0, alpha=0.7):
        """
        Soft Label Knowledge Distillation loss function.

        Args:
            temperature (float): Controls softening of logits (higher = softer).
            alpha (float): Balances between KD loss and true label loss.
        """
        super(SoftLabelDistillationLoss, self)._init_()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Computes the KD loss between student and teacher.

        Args:
            student_logits (Tensor): Output from student model.
            teacher_logits (Tensor): Output from teacher model (no grad).
            true_labels (Tensor): Ground-truth labels.

        Returns:
            loss (Tensor): Combined KD + CE loss.
        """
        T = self.temperature

        # Knowledge distillation loss (KL divergence between softened distributions)
        kd_loss = self.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T ** 2)

        # Regular cross-entropy loss with ground truth
        ce_loss = self.ce(student_logits, true_labels)

        # Weighted sum of both losses
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss