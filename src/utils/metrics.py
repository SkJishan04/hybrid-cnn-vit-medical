"""
Evaluation metrics. Macro-F1 and per-class metrics matter more than
accuracy here — HAM10000 is heavily imbalanced, so a model that always
predicts "nv" scores ~67% accuracy while being clinically useless.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate(model, dataloader, device, class_names: list) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }

    # one-vs-rest AUC-ROC, requires all classes present in the eval set
    try:
        results["auc_roc_macro"] = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        results["auc_roc_macro"] = float("nan")

    # per-class breakdown — this is what tells you if the model
    # is quietly failing on melanoma while acing nevi
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    results["per_class_f1"] = {
        class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))
    }

    return results


def format_results(results: dict) -> str:
    lines = [
        f"Accuracy:        {results['accuracy']:.4f}",
        f"Precision(macro): {results['precision_macro']:.4f}",
        f"Recall(macro):    {results['recall_macro']:.4f}",
        f"F1(macro):         {results['f1_macro']:.4f}",
        f"AUC-ROC(macro):   {results['auc_roc_macro']:.4f}",
        "Per-class F1:",
    ]
    for cls, f1 in results["per_class_f1"].items():
        lines.append(f"  {cls}: {f1:.4f}")
    return "\n".join(lines)