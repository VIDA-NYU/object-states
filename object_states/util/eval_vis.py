import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import get_cfg
# from IPython import embed


# ---------------------------------------------------------------------------- #
#                                 Visualization                                #
# ---------------------------------------------------------------------------- #


def emb_plot(plot_dir, X, y, prefix='', n=3000):
    fname = f'{plot_dir}/{prefix}_proj.png'
    if os.path.isfile(fname): return
    print("creating emb plot", fname)
    # Create a TSNE embedding plot (optional)
    # tsne = TSNE(n_components=2)
    m = Isomap()
    i = np.random.choice(np.arange(len(X)), size=n)
    X, y = X[i], y[i]
    Z = m.fit_transform(X)
    print(Z.shape)
    plt.figure(figsize=(10, 8))
    for c in np.unique(y):
        plt.scatter(Z[y==c, 0], Z[y==c, 1], label=str(c), s=20, alpha=0.3)
    plt.legend()
    plt.title(f'Embedding Projection: {prefix}')
    pltsave(fname)


def emission_plot(plot_dir, X, y, classes, prefix='', video_ids=None, show_ypred=False):
    plt.figure(figsize=(10, 8))
    plt.imshow(X.T, cmap='cubehelix', aspect='auto')
    cs = {c: i for i, c in enumerate(classes)}
    for c in set(y) - set(cs):
        cs[c] = len(cs)
    plt.plot(np.array([cs[yi] for yi in y]), c='r')
    if show_ypred:
        plt.plot(np.argmax(X, axis=1), c='white', alpha=0.3)
    ic = range(len(classes))
    plt.yticks(ic, [classes[i] for i in ic])
    pltsave(f'{plot_dir}/{prefix}emissions.png')
    np.savez(
        f'{plot_dir}/{prefix}emissions.npz', 
        predictions=X, ground_truth=y, 
        video_ids=video_ids, classes=classes)


def cm_plot(plot_dir, y_test, y_pred, classes, prefix=''):
    # classes = np.unique(y_test) if classes is None else classes
    cm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')*100
    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='magma', cbar=False, square=True,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix')
    pltsave(f'{plot_dir}/{prefix}confusion_matrix.png')


def n_videos_metrics(plot_dir, all_metrics, prefix=''):
    # Plot accuracy and F1-score vs. the number of videos
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_metrics.n_videos, all_metrics.accuracy, marker='o')
    plt.title('Accuracy vs. Number of Videos')
    plt.xlabel('Number of Videos')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(all_metrics.n_videos, all_metrics.f1, marker='o', color='orange')
    plt.title('F1 Score vs. Number of Videos')
    plt.xlabel('Number of Videos')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    pltsave(f'{plot_dir}/{prefix}accuracy_f1_vs_videos.png')


def n_videos_class_metrics(plot_dir, all_metrics, prefix=''):
    # Plot accuracy and F1-score vs. the number of videos
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for c, df in all_metrics.groupby('label'):
        # cc = df.class_count.mean()
        plt.plot(df.n_videos, df.accuracy, marker='o', label=c)#f'{c} {cc:.0f}'
    plt.legend()
    plt.title('Accuracy vs. Number of Videos')
    plt.xlabel('Number of Videos')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    for c, df in all_metrics.groupby('label'):
        # cc = df.class_count.mean()
        plt.plot(df.n_videos, df.accuracy, marker='o', label=c)#f'{c} {cc:.0f}'
    plt.title('F1 Score vs. Number of Videos')
    plt.xlabel('Number of Videos')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    pltsave(f'{plot_dir}/{prefix}accuracy_f1_vs_videos_per_class.png')


def pltsave(fname):
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    plt.savefig(fname)
    plt.close()
