import json
import matplotlib.pyplot as plt

# Load the two JSON files
with (open('source_OAI_base_bs_ut_roc_values.json') as f1, open('target_OAI_mmd_bs_ut_roc_values.json') as f2,
      open('target_OAI_base_bs_ut_roc_values.json') as f3, open('source_OAI_base_bs_ut_target_roc_values.json') as f4):
# with open('source_OAI_base_both_balanced_roc_values.json') as f1, open('target_OAI_base_both_balanced_roc_values.json') as f2:
    roc1 = json.load(f1)
    roc2 = json.load(f2)
    roc3 = json.load(f3)
    roc4 = json.load(f4)

# Extract values
fpr1, tpr1, auc1 = roc1['fpr'], roc1['tpr'], roc1['auc']
fpr2, tpr2, auc2 = roc2['fpr'], roc2['tpr'], roc2['auc']
fpr3, tpr3, auc3 = roc3['fpr'], roc3['tpr'], roc3['auc']
fpr4, tpr4, auc4 = roc4['fpr'], roc4['tpr'], roc4['auc']

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'trained+tested on source (AUC = {auc1:.3f})', color='blue')
plt.plot(fpr2, tpr2, label=f'mmd (AUC = {auc2:.3f})', color='green')
plt.plot(fpr3, tpr3, label=f'base (AUC = {auc3:.3f})', color='red')
plt.plot(fpr4, tpr4, label=f'trained+tested on target (AUC = {auc4:.3f})', color='black')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Unbalanced source, balanced target')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
