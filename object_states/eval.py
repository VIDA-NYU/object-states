import os
import glob
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
# from sklearn.svm import SVC, LinearSVC

# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap

from .config import get_cfg
from .util.step_annotations import load_object_annotations, get_obj_anns

from IPython import embed

import warnings
warnings.simplefilter('once')


STATE = 'state'


def remap_labels(sdf, old_col, new_col):
    RENAME = {
        '[partial]': '',
        '[full]': '',
        'floss-underneath': 'ends-cut',
        'floss-crossed': 'ends-cut',
        'raisins[cooked]': 'raisins',
        'oatmeal[cooked]+raisins': 'oatmeal+raisins',
        'teabag': 'tea-bag',
        '+stirrer': '',
        '[stirred]': '',
        'water+honey': 'water',
        'with-quesadilla': 'with-food',
        'with-pinwheels': 'with-food',
    }
    sdf[new_col] = sdf[old_col].copy()
    for old, new in RENAME.items():
        sdf[new_col] = sdf[new_col].str.replace(old, new)
    sdf = sdf[~sdf[new_col].isin(['folding', 'on-plate', 'rolling'])]
    return sdf


# ---------------------------------------------------------------------------- #
#                                 Data Loading                                 #
# ---------------------------------------------------------------------------- #

class bc:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_data(cfg, data_file_pattern, include=None):
    '''Load npz files (one per video) with embedding and label keys and concatenate
    
    
    
    '''
    # if os.path.isfile('dataset.pkl'):
    #     print('reading pickle')
    #     df = pd.read_pickle('dataset.pkl')
    #     print(df.head())
    #     return df
    use_aug = cfg.EVAL.USE_AUGMENTATIONS

    embeddings_list, df_list = [], []

    class_map = {}

    # steps_df, meta_df, object_names = load_annotations(cfg)
    dfs = load_object_annotations(cfg)

    fs = glob.glob(data_file_pattern)
    # if cfg.EVAL.TRAIN_BASE_ROOT:
    #     fs += glob.glob(f'{cfg.EVAL.TRAIN_BASE_ROOT}/embeddings/{cfg.EVAL.DETECTION_NAME}/*/clip/*.npz')
    if len(set(fs)) < len(fs):
        print("Warning duplicate files in training set!\n\n")
        input()

    print(f"Found {len(fs)} files", fs[:1])
    for f in tqdm.tqdm(fs, desc='loading data...'):
        if 'coffee_mit-eval' in f:
            embed()
        if include and not any(fi in f for fi in include):
            print("Skipping", f)
            continue
        data = np.load(f)
        z = data['z']
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        frame_idx = data['frame_index']

        # maybe filter out augmentations
        aug = data.get('augmented')
        if aug is None or use_aug:
            aug = np.zeros(len(z), dtype=bool)
        z = z[~aug]
        frame_idx = frame_idx[~aug]

        # get video ID and track ID
        video_id = data.get('video_name')
        if video_id is None:
            video_id = f.split('/')[-3]
        else:
            video_id = video_id.item()
        video_id = os.path.splitext(video_id)[0]
        track_id = data.get('track_id')
        if track_id is None:
            track_id = f.split('/')[-1].split('.')[0]
        else:
            track_id = track_id.item()
        track_id = int(track_id)

        if video_id not in dfs or track_id not in dfs[video_id]:
            tqdm.tqdm.write(f"{bc.FAIL}Skipping{bc.END}: {video_id}: {track_id}")
            continue
        tqdm.tqdm.write(f"Using: {video_id}: {track_id}")

        # get object state annotations
        ann = get_obj_anns(remap_labels(dfs[video_id][track_id], 'state', 'state'), frame_idx)
        embeddings_list.append(z)
        df_list.append(pd.DataFrame({
            'index': frame_idx,
            'object': ann.object,
            'state': ann.state,
            'track_id': track_id,
            'video_id': video_id,
        }))
        # print()
        # print(df_list[-1][['object', 'state']].value_counts())
        # print()
        # if input(): embed()

    X = np.concatenate(embeddings_list)
    df = pd.concat(df_list)
    df['vector'] = list(X)

    df.to_pickle('dataset.pkl')
    return df


def load_data_from_db(cfg, state_col, emb_type='clip'):
    import lancedb
    dfs = []
    fs = cfg.EVAL.EMBEDDING_DBS
    f = os.path.join(cfg.DATASET.ROOT, f'{emb_type}.lancedb')
    if not fs and os.path.isfile(f):
        fs = [f]
    for db_fname in fs:
        print(db_fname)
        assert os.path.isdir(db_fname)
        db = lancedb.connect(db_fname)
        for object_name in tqdm.tqdm(db.table_names()):
            dfs.append(db.open_table(object_name).to_pandas())
    df = pd.concat(dfs) if dfs else pd.DataFrame({state_col: []})
    if state_col:
        df['state'] = df[state_col]
    return df


def read_split_file(fname):
    lines = open(fname).read().splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith('#')]
    return lines


# ---------------------------------------------------------------------------- #
#                                   Training                                   #
# ---------------------------------------------------------------------------- #


def train_eval(run_name, model, X, y, i_train, i_test, video_ids, plot_dir='plots', **meta):
    '''Train and evaluate a model'''
    print(run_name, model)
    # plot_dir = f'{plot_dir}/{run_name}'
    # os.makedirs(plot_dir, exist_ok=True)

    X_train, X_test = X[i_train], X[i_test]
    y_train, y_test = y[i_train], y[i_test]
    # print(set(y_train))
    # print(set(y_test))
    # if input(): embed()

    # from imblearn.over_sampling import SMOTE
    # from imblearn.under_sampling import RandomUnderSampler
    # from imblearn.pipeline import Pipeline

    # # Create a pipeline to balance the classes using SMOTE
    # pipeline = Pipeline([
    #     # ('oversample', SMOTE(sampling_strategy='auto')),  # You can adjust sampling_strategy
    #     ('undersample', RandomUnderSampler(sampling_strategy='auto'))  # You can adjust sampling_strategy
    # ])

    # X_test2, y_test2 = X_test, y_test
    # X_test, y_test = pipeline.fit_resample(X_test, y_test)
    # print(X_test2.shape, y_test2.shape, X_test.shape, y_test.shape)

    assert not (set(video_ids[i_train]) & set(video_ids[i_test])), "Being extra sure... this is a nono"

    # Standardize features
    scaler = StandardScaler()
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ----------------------------------- Train ---------------------------------- #

    # Train the classifier
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_emis = model.predict_proba(X_test)

    # with open(os.path.join(plot_dir, f'{run_name}.pkl'), 'rb') as f:
    #     pickle.dump(y_pred, f)
    # Save the entire pipeline
    with open(os.path.join(plot_dir, f'{run_name}_pipeline.pkl'), 'wb') as f:
        joblib.dump(pipeline, f)

    # ------------------------------- Visualization ------------------------------ #

    # Generate plots

    all_metrics = []

    # compute vanilla metrics
    meta['run_name'] = meta['metric_name'] = run_name
    metrics = get_metrics(y_test, y_pred, **meta)
    all_metrics.append(metrics)
    tqdm.tqdm.write(f'Accuracy for {run_name}: {metrics["accuracy"]:.2f}')

    # generate vanilla plots
    video_ids_test = video_ids[i_test]
    emission_plot(plot_dir, y_emis, y_test, model.classes_, f'{run_name}_ma0_', video_ids=video_ids_test)
    emission_plot(plot_dir, y_emis, y_test, model.classes_, f'{run_name}_ma0_ypred_', show_ypred=True, video_ids=video_ids_test)
    cm_plot(plot_dir, y_test, y_pred, model.classes_, f'{run_name}_')

    # with moving average
    for winsize in [2, 4, 8, 16]:
        y_ = moving_average(y_emis, winsize)
        y_pred_ = np.asarray(model.classes_)[np.argmax(y_, axis=1)]
        # emission_plot(plot_dir, y_, y_test, model.classes_, f'{run_name}_ma{winsize}_', video_ids=video_ids_test)
        emission_plot(plot_dir, y_, y_test, model.classes_, f'{run_name}_ma{winsize}_ypred_', show_ypred=True, video_ids=video_ids_test)
        cm_plot(plot_dir, y_test, y_pred_, model.classes_, f'{run_name}_cm_ma{winsize}_')

        meta = {**meta}
        meta['metric_name'] = f'{run_name}_movingavg-{winsize}'
        metrics = get_metrics(y_test, y_pred_, smoothing='ma', win_size=winsize, **meta)
        all_metrics.append(metrics)

    for alpha in [0.1, 0.2, 0.5]:
        y_ = exponentially_decaying_average(y_emis, alpha)
        y_pred_ = np.asarray(model.classes_)[np.argmax(y_, axis=1)]
        emission_plot(plot_dir, y_, y_test, model.classes_, f'{run_name}_ema{alpha}_ypred_', show_ypred=True, video_ids=video_ids_test)
        cm_plot(plot_dir, y_test, y_pred, model.classes_, f'{run_name}_')

        meta = {**meta}
        meta['metric_name'] = f'{run_name}_expmovingavg-{alpha}'
        metrics = get_metrics(y_test, y_pred_, smoothing='ema', alpha=alpha, **meta)
        all_metrics.append(metrics)

    # y_hmm = hmm_forward(y_emis, len(model.classes_))
    # emission_plot(plot_dir, y_hmm, y_test, model.classes_, f'{run_name}_trans_', video_ids=video_ids_test)
    # emission_plot(plot_dir, y_hmm, y_test, model.classes_, f'{run_name}_trans_ypred_', show_ypred=True, video_ids=video_ids_test)
    # # embed()
    
    # get per class metrics
    per_class_metrics = []
    for c in np.unique(y):
        per_class_metrics.append(get_metrics(
            y_test[y_test==c], y_pred[y_test==c], label=c, **meta))

    
    # tqdm.tqdm.write(f'F1 for {run_name}: {metrics["f1"]:.2f}')
    return all_metrics, per_class_metrics


def get_metrics(y_test, y_pred, **meta):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=np.nan, average='macro')
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score,
        'ap': precision,
        'avg_recal': recall,
        **meta
    }


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
    classes = list(classes)
    for c in set(y) - set(cs):
        cs[c] = len(cs)
        classes.append(c)
    plt.plot(np.array([cs[yi] for yi in y]), c='r')
    if show_ypred:
        plt.scatter(np.arange(len(X)), np.argmax(X, axis=1), c='white', s=5, alpha=0.2)
    ic = range(len(classes))
    plt.yticks(ic, [classes[i] for i in ic])
    pltsave(f'{plot_dir}/{prefix}emissions.png')
    os.makedirs(f'{plot_dir}/npzs', exist_ok=True)
    np.savez(
        f'{plot_dir}/npzs/{prefix}emissions.npz', 
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


def cross_model_metrics(plot_dir, all_metrics, prefix=''):
    # Plot accuracy and F1-score vs. the number of videos
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for name, mdf in all_metrics[all_metrics.smoothing == 'ma'].groupby("run_name"):
        plt.plot(mdf.win_size, mdf.f1, label=name)
    plt.legend()

    plt.title('F1 Score vs. Number of Videos')
    plt.xlabel('Moving Average Window Size')
    plt.ylabel('F1 Score')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    for name, mdf in all_metrics[all_metrics.smoothing == 'ema'].groupby("run_name"):
        plt.plot(mdf.alpha, mdf.f1, label=name)
    plt.legend()

    plt.title('F1 Score vs. EMA alpha * x[t] + (1 - alpha) * x[t-1]')
    plt.xlabel('Exp Moving Average alpha')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    pltsave(f'{plot_dir}/{prefix}accuracy_f1_vs_smooth.png')


# def n_videos_class_metrics(plot_dir, all_metrics, prefix=''):
#     # Plot accuracy and F1-score vs. the number of videos
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     for c, df in all_metrics.groupby('label'):
#         # cc = df.class_count.mean()
#         plt.plot(df.n_videos, df.accuracy, marker='o', label=c)#f'{c} {cc:.0f}'
#     plt.legend()
#     plt.title('Accuracy vs. Number of Videos')
#     plt.xlabel('Number of Videos')
#     plt.ylabel('Accuracy')

#     plt.subplot(1, 2, 2)
#     for c, df in all_metrics.groupby('label'):
#         # cc = df.class_count.mean()
#         plt.plot(df.n_videos, df.accuracy, marker='o', label=c)#f'{c} {cc:.0f}'
#     plt.title('F1 Score vs. Number of Videos')
#     plt.xlabel('Number of Videos')
#     plt.ylabel('F1 Score')
#     plt.legend()
#     plt.tight_layout()
#     pltsave(f'{plot_dir}/{prefix}accuracy_f1_vs_videos_per_class.png')


def pltsave(fname):
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    plt.savefig(fname)
    plt.close()


# ---------------------------------------------------------------------------- #
#                                      HMM                                     #
# ---------------------------------------------------------------------------- #

# def create_hmm(num_states, p_self=0.9):
#     transition_matrix = np.eye(num_states) * p_self + (1.0 - p_self) / (num_states - 1)
#     emission_matrix = np.eye(num_states)
#     initial_prob = np.ones(num_states) / num_states
#     return initial_prob, emission_matrix, transition_matrix

# Forward pass to compute the forward probabilities
def hmm_forward(sequence, num_states, p_self=0.9):
    transition_matrix = np.eye(num_states) * p_self + (1.0 - p_self) / (num_states - 1)
    forward_prob = np.zeros((len(sequence), num_states))
    forward_prob[0, :] = 1 / num_states
    for t in range(1, len(sequence)):
        for j in range(num_states):
            forward_prob[t] = np.sum(forward_prob[t - 1, i] * transition_matrix[i, j] for i in range(num_states))
            forward_prob[t] *= 1.0 / np.sum(forward_prob[t])
    return forward_prob



def moving_average(a, n=3, axis=0):
    ret = np.cumsum(a, dtype=float, axis=axis)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] = ret[:n] / np.arange(n)[:, None]
    return ret


def exponentially_decaying_average(a, decay_rate):
    assert 0 < decay_rate < 1, "Decay rate must be between 0 and 1."
    result = a.copy()
    result[0, :] = a[0, :]
    for t in range(1, a.shape[0]):
        result[t, :] = decay_rate * result[t - 1, :] + (1 - decay_rate) * a[t, :]
    return result

# ---------------------------------------------------------------------------- #
#                              Training Meta Loop                              #
# ---------------------------------------------------------------------------- #




def get_data(cfg, STATE, full_split, emb_type='clip'):
    emb_dirs = cfg.EVAL.EMBEDDING_DIRS or [os.path.join(cfg.DATASET.ROOT, 'embeddings-all', cfg.EVAL.DETECTION_NAME)]
    ydf = load_data_from_db(cfg, state_col='mod_state')
    db_train_split = ydf.video_id.unique().tolist()
    ydf = pd.concat([
        *[
            load_data(cfg, f'{d}/{cfg.EVAL.DETECTION_NAME}/*/{emb_type}/*.npz', include=set(full_split) - set(ydf.video_id.unique()))
            for d in emb_dirs
        ],
        ydf
    ])
    print(ydf.groupby('object').state.value_counts())

    # sampling 12k per state
    ydf = sample_random(ydf, STATE, 15000)
    print(ydf.groupby('object').state.value_counts())
    print('Nulls:', ydf[pd.isna(ydf.state)].video_id.value_counts())
    assert None not in set(ydf.state)
    return ydf, db_train_split


def sample_random(df, STATE, n):
    df = df.groupby(STATE, group_keys=False).apply(lambda x: x.sample(min(len(x), n)))
    return df


def get_models(cfg):
    return [
        # (KNeighborsClassifier, 'knn5',  {'n_neighbors': 5}),
        # (KNeighborsClassifier, 'knn11-50', {'n_neighbors': 11}, lambda df: sample_random(df, STATE, 50)),
        # (KNeighborsClassifier, 'knn11-100', {'n_neighbors': 11}, lambda df: sample_random(df, STATE, 100)),
        (KNeighborsClassifier, 'knn5-2000', {'n_neighbors': 5}, lambda df: sample_random(df, STATE, 2000)),
        (KNeighborsClassifier, 'knn21-2000', {'n_neighbors': 21}, lambda df: sample_random(df, STATE, 2000)),
        (KNeighborsClassifier, 'knn11-1000', {'n_neighbors': 11}, lambda df: sample_random(df, STATE, 1000)),
        (KNeighborsClassifier, 'knn11-2000', {'n_neighbors': 11}, lambda df: sample_random(df, STATE, 2000)),
        (KNeighborsClassifier, 'knn11-5000', {'n_neighbors': 11}, lambda df: sample_random(df, STATE, 5000)),
        (KNeighborsClassifier, 'knn11-12000', {'n_neighbors': 11}, lambda df: sample_random(df, STATE, 12000)),
        # (KNeighborsClassifier, 'knn50', {'n_neighbors': 50}),
        (LogisticRegression, 'logreg',  {}, lambda df: sample_random(df, STATE, 5000)),
        (RandomForestClassifier, 'rf',  {}, lambda df: sample_random(df, STATE, 5000)),
        # (
        #     make_pipeline(
        #         StandardScaler(),
        #         KBinsDiscretizer(encode="onehot", random_state=0),
        #         LogisticRegression(random_state=0),
        #     ),
        #     'kbins_logreg', 
        #     {
        #         "kbinsdiscretizer__n_bins": np.arange(5, 8),
        #         "logisticregression__C": np.logspace(-1, 1, 3),
        #     },
        # ),
    ]


def prepare_data(odf, STATE, sampler, train_split, val_split):
    video_ids = odf['video_id'].values
    unique_video_ids = np.unique(video_ids)

    # obj_train_base_split = [f for f in train_base_split if f in video_ids and f not in val_split]
    obj_train_split = [f for f in train_split if f in video_ids and f not in val_split]
    obj_val_split = [f for f in val_split if f in video_ids]
    # embed()
    # obj_train_split = sorted(obj_train_split, key=lambda v: -len(odf[odf.video_id == v].state.unique()))

    # print("Base Training split:", obj_train_base_split)
    print("Training split:", obj_train_split)
    print("Validation split:", obj_val_split)
    print("Unused videos:", set(unique_video_ids) - set(obj_train_split+obj_val_split))
    print("Missing videos:", set(obj_train_split+obj_val_split) - set(unique_video_ids))

    print()
    print("all data:")
    print('X', X.shape)
    print('y', y.shape)
    print(odf[['video_id', 'track_id', STATE]].value_counts())
    i_train = np.isin(video_ids, obj_train_split)
    # i_train = np.isin(video_ids, obj_train_base_split + obj_train_split[:nvids])
    i_val = np.isin(video_ids, obj_val_split)

    if sampler is not None:
        odf = pd.concat([sampler(odf.iloc[i_train]), odf.iloc[i_val]])
    X = np.array(list(odf['vector'].values))
    y = odf[STATE].values
    return X, y, video_ids, i_train, i_val


import ipdb
@ipdb.iex
def run(config_name):
    cfg = get_cfg(config_name)
    root_plot_dir = root_plot_dir_ = cfg.EVAL.PLOT_DIR or 'plots'
    # i=0
    # while os.path.isdir(root_plot_dir_):
    #     root_plot_dir_ = root_plot_dir + f'_{i}'
    #     i+=1
    # root_plot_dir=root_plot_dir_
    if os.path.isdir(root_plot_dir):
        raise RuntimeError(f"{root_plot_dir} exists")
    os.makedirs(root_plot_dir, exist_ok=True)

    # STATE = 'super_simple_state'
    # STATE = 'state'

    train_split = read_split_file(cfg.EVAL.TRAIN_CSV)
    train_base_split = read_split_file(cfg.EVAL.TRAIN_BASE_CSV)
    val_splits = [(f, read_split_file(f)) for f in cfg.EVAL.VAL_CSVS]
    print(len(train_base_split), train_base_split[:5])
    print(len(train_split), train_split[:5])
    print(len(val_splits), val_splits[:5])

    full_train_split = train_split + train_base_split
    full_val_split = [x for f, xs in val_splits for x in xs]
    full_split = full_train_split + full_val_split
    print(full_split)

    for _,val_split in val_splits:
        assert not set(full_train_split) & set(val_split), f"what are you doing silly {set(full_train_split) & set(val_split)}"


    models = get_models(cfg)

    cfg.EVAL.EMBEDDING_TYPES=['clip']
    for emb_type in tqdm.tqdm(cfg.EVAL.EMBEDDING_TYPES, desc='embedding type'):
        ydf, db_train_split = get_data(cfg, STATE, full_split)
        emb_plot(f'{root_plot_dir}/{emb_type}', np.array(list(ydf['vector'].values)), ydf['object'].values, 'object')
        # emb_plot(f'{root_plot_dir}/{emb_type}', np.array(list(ydf['vector'].values)), ydf[STATE].values, 'states')

        for (val_split_fname, val_split) in val_splits:
            val_split_name = val_split_fname.split('/')[-1].removesuffix('.txt')
            for object_name, odf in ydf.groupby('object'):
                plot_dir = f'{root_plot_dir}/{val_split_name}/{emb_type}/{object_name}'
                os.makedirs(plot_dir, exist_ok=True)

                all_metrics = []
                all_per_class_metrics = []

                for cls, name, kw, sampler in tqdm.tqdm(models, desc='models'):
                    X, y, video_ids, i_train, i_val = prepare_data(odf, STATE, sampler, db_train_split, val_split)
                    emb_plot(f'{root_plot_dir}/{emb_type}_{object_name}', X, y, 'states')

                    if not i_val.sum():
                        print(f"\n\n\n\nSKIPPING i_val is empty. {val_split}\n\n\n")
                        continue

                    print(f"Training with: train size: {len(i_train)} val size: {len(i_val)}")

                    print("Train Counts:")
                    train_counts = show_counts(y[i_train])
                    print(train_counts)
                    print("Val Counts:")
                    val_counts = show_counts(y[i_val])
                    print(val_counts)
                
                    model = cls(**kw)
                    metrics, per_class_metrics = train_eval(
                        f'{val_split_name}_{emb_type}_{name}', model, 
                        X, y, i_train, i_val, 
                        video_ids=video_ids,
                        plot_dir=plot_dir,
                        model_name=name,
                        # n_videos=nvids,
                        **kw)
                    all_metrics.extend(metrics)
                    all_per_class_metrics.extend(per_class_metrics)

                all_metrics_df = pd.DataFrame(all_metrics)
                all_per_class_metrics_df = pd.DataFrame(all_per_class_metrics)

                all_metrics_df.to_csv(f'{plot_dir}/metrics.csv')
                all_per_class_metrics_df.to_csv(f'{plot_dir}/class_metrics.csv')

                # ---------- Show how it performs as a function of number of videos ---------- #

                if len(all_metrics):
                    cross_model_metrics(plot_dir, all_metrics_df, f'{emb_type}_')
                    # n_videos_class_metrics(plot_dir, all_per_class_metrics_df, f'{emb_type}_')


def show_counts(y):
    yu, counts = np.unique(y, return_counts=True)
    for yui, c in zip(yu, counts):
        print(yui, c)
    return dict(zip(yu, counts))

import ipdb
@ipdb.iex
def show_data(config_name, emb_type='clip'):
    cfg = get_cfg(config_name)
    emb_dir = os.path.join(cfg.DATASET.ROOT, 'embeddings1', cfg.EVAL.DETECTION_NAME)
    emb_types = cfg.EVAL.EMBEDDING_TYPES
    data_file_pattern = f'{emb_dir}/*/{emb_type}/*.npz'
    # dfs = load_object_annotations(cfg)
    # # for vid, odfs in dfs.items():
    # #     print(vid)
    # #     print(set(odfs))
    # # input()
    # for vid, odfs in dfs.items():
    #     print(vid)
    #     print({k: odfs[k].shape for k in odfs})
    # for vid, odfs in dfs.items():
    #     print(vid)
    #     for k in odfs:
    #         print(k)
    #         print(odfs[k])
    # embed()

    # X, y, video_ids, class_map = load_data(cfg, data_file_pattern, use_aug=False)
    # df = pd.DataFrame({'vids': video_ids, 'y': y})
    # df['label'] = df.y.apply(lambda y: class_map[y])
    # for v, rows in df.groupby('vids'):
    #     print(v)
    #     print(rows.label.value_counts())

    df = load_data(cfg, data_file_pattern, use_aug=False)
    for object_name, odf in df.groupby('object'):
        print(object_name)
        print(odf[['state']].value_counts())
        # print(odf[['state', 'video_id']].value_counts())

        x = odf[['state', 'video_id']].value_counts().unstack().fillna(0)
        print(x)
        x.to_csv(f"{object_name}_video_counts.csv")
    # embed()

if __name__ == '__main__':
    import fire
    fire.Fire()