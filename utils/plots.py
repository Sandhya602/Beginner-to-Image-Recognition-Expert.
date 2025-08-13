import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_learning_curves(history, out_path):
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy vs Epochs')
    plt.savefig(out_path, bbox_inches='tight'); plt.close()

def plot_confusion(y_true, y_pred, out_path, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight'); plt.close()
  
