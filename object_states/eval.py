import os
import glob
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap

from .config import get_cfg

from IPython import embed


# ---------------------------------------------------------------------------- #
#                                 Data Loading                                 #
# ---------------------------------------------------------------------------- #


def load_object_annotations(cfg):
    meta_df = pd.read_csv(cfg.DATASET.META_CSV).set_index('video_name')
    object_names = []
    for c in meta_df.columns:
        if c.startswith('#'):
            meta_df[c[1:]] = meta_df.pop(c).fillna('').apply(lambda s: [int(float(x)) for x in str(s).split('+') if x != ''])
            object_names.append(c[1:])

    states_df = pd.read_csv(cfg.DATASET.STATES_CSV)
    states_df = states_df[states_df.time.fillna('') != '']
    states_df['time'] = pd.to_timedelta(states_df.time.apply(lambda x: f'00:{x}'))
    states_df['start_frame'] = (states_df.time.dt.total_seconds() * meta_df.fps.loc[states_df.video_name].values).astype(int)
    
    # creating a dict of {video_id: {track_id: df}}
    dfs = {}
    for vid, row in meta_df.iterrows():
        objs = {}
        sdf = states_df[states_df.video_name == vid]
        for c in object_names:
            if c not in sdf.columns:
                continue
            odf = sdf[[c, 'start_frame']].copy().rename(columns={c: "state"})
            odf = odf[odf.state.fillna('') != '']
            odf['stop_frame'] = odf['start_frame'].shift(-1)
            odf['object'] = c
            if not len(odf):
                continue
            for track_id in row[c]:
                objs[track_id] = odf
        # if not len(objs):
        #     continue
        dfs[vid] = objs
            
    return dfs

def get_obj_anns(df, frame_idx):
    idxs = []
    for i in frame_idx:
        d = df[(df.start_frame <= i) & (pd.isna(df.stop_frame) | (df.stop_frame > i))]
        idxs.append({
            'state': d.state.iloc[0] if len(d) else None,
            'object': d.object.iloc[0] if len(d) else None,
        })
    return pd.DataFrame(idxs)


def load_data(cfg, data_file_pattern, use_aug=True):
    '''Load npz files (one per video) with embedding and label keys and concatenate
    
    
    
    '''
    embeddings_list, df_list = [], []

    class_map = {}

    # steps_df, meta_df, object_names = load_annotations(cfg)
    dfs = load_object_annotations(cfg)

    fs = glob.glob(data_file_pattern)
    print(f"Found {len(fs)} files", fs[:1])
    for f in tqdm.tqdm(fs, desc='loading data...'):
        data = np.load(f)
        z = data['z']
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
        video_id = video_id.removesuffix('.mp4')
        track_id = data.get('track_id')
        if track_id is None:
            track_id = f.split('/')[-1].split('.')[0]
        else:
            track_id = track_id.item()
        track_id = int(track_id)

        if video_id not in dfs or track_id not in dfs[video_id]:
            tqdm.tqdm.write(f"Skipping: {video_id}: {track_id}")
            continue
        tqdm.tqdm.write(f"Using: {video_id}: {track_id}")

        # get object state annotations
        ann = get_obj_anns(dfs[video_id][track_id], frame_idx)
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
    df['Z'] = list(X)
    return df


def read_split_file(fname):
    lines = open(fname).read().splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith('#')]
    return lines


# ---------------------------------------------------------------------------- #
#                                   Training                                   #
# ---------------------------------------------------------------------------- #


def train_eval(run_name, model, X, y, i_train, i_test, plot_dir='plots', **meta):
    '''Train and evaluate a model'''
    # plot_dir = f'{plot_dir}/{run_name}'
    # os.makedirs(plot_dir, exist_ok=True)

    X_train, X_test = X[i_train], X[i_test]
    y_train, y_test = y[i_train], y[i_test]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ----------------------------------- Train ---------------------------------- #

    # Train the classifier
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # ------------------------------- Visualization ------------------------------ #

    # Generate plots
    
    cm_plot(plot_dir, y_test, y_pred, model.classes_, f'{run_name}_')

    # ---------------------------------- Metrics --------------------------------- #

    # Compile metrics
    meta['run_name'] = run_name
    metrics = get_metrics(y_test, y_pred, **meta)

    per_class_metrics = []
    for c in np.unique(y):
        per_class_metrics.append(get_metrics(
            y_test[y_test==c], y_pred[y_test==c], label=c, **meta))

    tqdm.tqdm.write(f'Accuracy for {run_name}: {metrics["accuracy"]:.2f}')
    # tqdm.tqdm.write(f'F1 for {run_name}: {metrics["f1"]:.2f}')
    return metrics, per_class_metrics


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


# ---------------------------------------------------------------------------- #
#                              Training Meta Loop                              #
# ---------------------------------------------------------------------------- #


import ipdb
@ipdb.iex
def run(config_name):
    cfg = get_cfg(config_name)
    os.makedirs('plots', exist_ok=True)

    # emb_dir = cfg.DATASET.EMBEDDING_DIR
    emb_dir = os.path.join(cfg.DATASET.ROOT, 'embeddings1', cfg.EVAL.DETECTION_NAME)
    emb_types = cfg.EVAL.EMBEDDING_TYPES

    train_split = read_split_file(cfg.EVAL.TRAIN_CSV)
    val_split = read_split_file(cfg.EVAL.VAL_CSV)

    assert not set(train_split) & set(val_split), f"what are you doing silly {set(train_split) & set(val_split)}"

    all_metrics = []
    all_per_class_metrics = []

    models = [
        # (KNeighborsClassifier, 'knn5',  {'n_neighbors': 5}),
        (KNeighborsClassifier, 'knn11', {'n_neighbors': 11}),
        # (KNeighborsClassifier, 'knn50', {'n_neighbors': 50}),
        # (RandomForestClassifier, 'rf',  {}),
    ]

    # , 'detic', 'detic_s0', 'detic_s1', 'detic_s2'
    emb_types=['clip']
    for emb_type in tqdm.tqdm(emb_types, desc='embedding type'):
        data_file_pattern = f'{emb_dir}/*/{emb_type}/*.npz'

        ydf = load_data(cfg, data_file_pattern, use_aug=True)
        emb_plot(f'plots/{emb_type}', np.array(list(ydf['Z'].values)), ydf['object'].values, 'object')

        for object_name, odf in ydf.groupby('object'):
            if object_name!='tortilla': continue
            X = np.array(list(odf['Z'].values))
            y = odf['state'].values
            video_ids = odf['video_id'].values

            obj_train_split = [f for f in train_split if f in video_ids]
            obj_val_split = [f for f in val_split if f in video_ids]
            # obj_train_split = sorted(obj_train_split, key=lambda v: -len(odf[odf.video_id == v].state.unique()))

            print()
            print("all data:")
            print('X', X.shape)
            print('y', y.shape)
            print(object_name)
            print(X.shape)
            print(odf[['video_id', 'track_id', 'state']].value_counts())

            plot_dir = f'plots/{emb_type}/{object_name}'
            os.makedirs(plot_dir, exist_ok=True)
            # emb_plot(plot_dir, X, y, 'state')
            # emb_plot(plot_dir, X, video_ids, 'video_id')

            all_metrics = []
            all_per_class_metrics = []

            for nvids in tqdm.tqdm(range(len(obj_train_split)), desc='n videos'):
                nvids += 1
                i_train = np.isin(video_ids, obj_train_split[:nvids])
                i_val = np.isin(video_ids, obj_val_split)

                a=odf.iloc[i_train][['video_id', 'state']].value_counts()
                a.to_csv(f"{plot_dir}/train_stats_{nvids}vids.csv")
                print(a)
                a=odf.iloc[i_val][['video_id', 'state']].value_counts()
                a.to_csv(f"{plot_dir}/val_stats_{nvids}vids.csv")
                print(a)

                x = np.unique(y[i_train])

                print(f"Training with {nvids}: train size: {len(i_train)} val size: {len(i_val)}")

                print("Train Counts:")
                train_counts = show_counts(y[i_train])
                print("Val Counts:")
                val_counts = show_counts(y[i_val])


                for cls, name, kw in tqdm.tqdm(models, desc='models'):
                    model = cls(**kw)
                    metrics, per_class_metrics = train_eval(
                        f'{emb_type}_{name}_{nvids}vid', model, 
                        X, y, i_train, i_val, 
                        plot_dir=plot_dir,
                        model_name=name,
                        n_videos=nvids,
                        **kw)
                    all_metrics.append(metrics)
                    all_per_class_metrics.extend(per_class_metrics)
                    # for d in per_class_metrics:
                    #     d['class_count'] = train_counts[d['label']]

            all_metrics_df = pd.DataFrame(all_metrics)
            all_per_class_metrics_df = pd.DataFrame(all_per_class_metrics)

            all_metrics_df.to_csv(f'{plot_dir}/metrics.csv')
            all_per_class_metrics_df.to_csv(f'{plot_dir}/class_metrics.csv')

            # ---------- Show how it performs as a function of number of videos ---------- #

            n_videos_metrics(plot_dir, all_metrics_df, f'{emb_type}_')
            n_videos_class_metrics(plot_dir, all_per_class_metrics_df, f'{emb_type}_')

            for n in all_metrics_df.model_name.unique():
                n_videos_metrics(plot_dir, all_metrics_df[all_metrics_df.model_name == n], f'{n}_')
                n_videos_class_metrics(plot_dir, all_per_class_metrics_df[all_per_class_metrics_df.model_name == n], f'{n}_')


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