import json
import matplotlib.pyplot as plt

# Load the two JSON files
with (open('target_CHECK_base_roc_values.json') as f1, open('source_transformed_CHECK_base_roc_values.json') as f2,
      open('target_check_mmd_both_balanced_roc_values.json') as f3, open('target_check_mmd_both_unbalanced_roc_values.json') as f4,
      open('target_check_mmd_both_unbalanced_inverted_roc_values.json') as f5, open('target_check_mmd_bs_ut_roc_values.json') as f6,
      open('target_check_mmd_both_balanced_roc_values.json') as f7):
    # with open('source_OAI_base_both_balanced_roc_values.json') as f1,
    # open('target_OAI_base_both_balanced_roc_values.json') as f2:
    roc1 = json.load(f1)
    roc2 = json.load(f2)
    roc3 = json.load(f3)
    roc4 = json.load(f4)
    roc5 = json.load(f5)
    roc6 = json.load(f6)
    roc7 = json.load(f7)
    # roc5 = json.load(f5)

# Extract values
fpr1, tpr1, auc1 = roc1['fpr'], roc1['tpr'], roc1['auc']
fpr2, tpr2, auc2 = roc2['fpr'], roc2['tpr'], roc2['auc']
fpr3, tpr3, auc3 = roc3['fpr'], roc3['tpr'], roc3['auc']
fpr4, tpr4, auc4 = roc4['fpr'], roc4['tpr'], roc4['auc']
fpr5, tpr5, auc5 = roc5['fpr'], roc5['tpr'], roc5['auc']
fpr6, tpr6, auc6 = roc6['fpr'], roc6['tpr'], roc6['auc']
fpr7, tpr7, auc7 = roc7['fpr'], roc7['tpr'], roc7['auc']
# fpr5, tpr5, auc5 = roc5['fpr'], roc5['tpr'], roc5['auc']


# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'Trained on normal tested on transformed (AUC = {auc1:.3f})', color='blue')
plt.plot(fpr2, tpr2, label=f'Trained + tested on transformed (AUC = {auc2:.3f})', color='green')
plt.plot(fpr3, tpr3, label=f'50/50 v 50/50 (AUC = {auc3:.3f})', color='red')
plt.plot(fpr4, tpr4, label=f'70/30 v 70/30 (AUC = {auc4:.3f})', color='purple')
plt.plot(fpr5, tpr5, label=f'70/30 v 30/70 (AUC = {auc5:.3f})', color='black')
plt.plot(fpr6, tpr6, label=f'50/50 v 70/30 (AUC = {auc6:.3f})', color='pink')
plt.plot(fpr7, tpr7, label=f'70/30 v 50/50 (AUC = {auc7:.3f})', color='orange')


plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MMD domain adaptation results, trained on normal, same training size')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
