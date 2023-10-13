import os
import glob
import tqdm
import pathtrees as pt
import numpy as np
import pandas as pd
import lancedb
import matplotlib.pyplot as plt

from IPython import embed

from ..config import get_cfg


def load_object_annotations(meta_csv, states_csv):
    meta_df = pd.read_csv(meta_csv).set_index('video_name').groupby(level=0).last()
    object_names = []
    for c in meta_df.columns:
        if c.startswith('#'):
            meta_df[c[1:]] = meta_df.pop(c).fillna('').apply(lambda s: [int(float(x)) for x in str(s).split('+') if x != ''])
            object_names.append(c[1:])

    states_df = pd.read_csv(states_csv)
    states_df = states_df[states_df.time.fillna('') != '']
    # print(states_df.shape)
    print(set(states_df.video_name.unique()) - set(meta_df.index.unique()))
    states_df = states_df[states_df.video_name.isin(meta_df.index.unique())]
    states_df['time'] = pd.to_timedelta(states_df.time.apply(lambda x: f'00:{x}'))
    states_df['start_frame'] = (states_df.time.dt.total_seconds() * meta_df.fps.loc[states_df.video_name].values).astype(int)
    # print(states_df.shape) 
    
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
            odf = odf.drop_duplicates(subset=['start_frame'], keep='last')
            odf['stop_frame'] = odf['start_frame'].shift(-1)
            odf['object'] = c
            if not len(odf):
                continue
            for track_id in row[c]:
                objs[track_id] = odf
            print(vid, track_id, odf.shape)
        dfs[vid] = objs
            
    return dfs


def get_obj_anns(dfs, frame_idx):
    idxs = []
    for i in frame_idx:
        ds = {
            k: df[(df.start_frame <= i) & (pd.isna(df.stop_frame) | (df.stop_frame > i))]
            for k, df in dfs.items()
        }
        obj = list(set(d.object.iloc[0] for d in ds.values() if len(d)))
        assert len(obj) < 2, f"Something is wrong.. disagreeing labels ({obj}) assigned to object track"
            
        idxs.append({
            **{f'{k}_state': d.state.iloc[-1] for k, d in ds.items() if len(d)},
            'object': obj[0],
        } if obj else {})
    return pd.DataFrame(idxs)


def load_data(cfg, data_file_pattern, use_aug=True):
    '''Load npz files (one per video) with embedding and label keys and concatenate
    
    
    
    '''
    embeddings_list, df_list = [], []

    dfs = {
        k: load_object_annotations(cfg.DATASET.META_CSV, f)
        for k, f in cfg.DATASET.STATES_CSVS.items()
    }
    # embed()

    fs = glob.glob(data_file_pattern)
    print(f"Found {len(fs)} files", fs[:1])
    for f in tqdm.tqdm(fs, desc='loading data...'):
        # print(f)
        # if 'pinwheels' not in f and 'quesadilla' not in f: continue
        # if 'plain' not in f: 
        #     print(f)
        #     continue
        data = np.load(f)
        z = data['z'].astype(np.float32)
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        frame_idx = data['frame_index']

        # maybe filter out augmentations
        augmented = data.get('augmented')
        if augmented is None:
            augmented = np.zeros(len(z), dtype=bool)
        if not use_aug:
            z = z[~augmented]
            frame_idx = frame_idx[~augmented]

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

        dfsi = {k: df[video_id][track_id] for k, df in dfs.items() if video_id in df and track_id in df[video_id]}
        for k in dfsi:
            tqdm.tqdm.write(f"{k} {set(dfs[k])&{video_id}}")
            if video_id in dfs[k]:
                tqdm.tqdm.write(f"{k} {set(dfs[k][video_id])&{track_id}}")
        if not dfsi:
            tqdm.tqdm.write(f"Skipping: {video_id}: {track_id}")
            continue
        # tqdm.tqdm.write(f"Using: {video_id}: {track_id}")

        # get object state annotations
        ann = get_obj_anns(dfsi, frame_idx)
        if not all(ann.shape):
            tqdm.tqdm.write(f"no data for {video_id}.{track_id} {ann.shape} {z.shape}")
            for k in dfs:
                tqdm.tqdm.write(f"{k} {set(dfs[k])&{video_id}}")
            continue
        tqdm.tqdm.write(f"using {video_id}.{track_id} {ann.shape} {set(dfsi)} {z.shape}")
        embeddings_list.append(z)
        df_list.append(pd.DataFrame({
            'index': frame_idx,
            'object': ann.object,
            **{k: ann[k] for k in ann.columns if k.endswith('_state')},
            'track_id': track_id,
            'video_id': video_id,
            'augmented': augmented,
        }))
        # break

    df = pd.concat(df_list)
    df['vector'] = [x for xs in embeddings_list for x in xs]
    return df


def cluster_reduce(data, y, n_clusters=200, n_pts_per_cluster=20, n_clusters_per_class=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    embed()
    data = data[y=='peanut-butter']
    y=y[y=='peanut-butter']
    # Step 0: Normalization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Step 1: PCA
    print(data.shape)
    pca = PCA(n_components=30)  # Choose the number of components based on your data
    data_pca = pca.fit_transform(data)
    print(data_pca.shape)

    plt.figure(figsize=(15, 15))
    for yi in np.unique(y):
        plt.scatter(data_pca[y==yi, 0], data_pca[y==yi, 1], label=yi)
    plt.legend()
    plt.savefig("PCA.png")
    plt.close()

    # Perform k-means clustering
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    p = 1 / counts[inv]
    p = p / p.sum()
    n = n_clusters * n_pts_per_cluster * 10
    if n < len(y):
        idxs = np.random.choice(len(y), n, p=p, replace=False)
        np.sort(idxs)
        data = data[idxs]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_pca)
    clusters = kmeans.labels_
    # unique, count = np.unique(labels, return_counts=True)
    # for c, u in sorted(zip(count, unique)):
    #     print(u, c)

    # Sample a fixed number of points from each cluster
    sampled_points = []
    cluster_indices = {}
    for i in np.unique(clusters):
        indices = np.where(clusters == i)[0]
        y_counts = pd.Series(y[indices]).value_counts()
        y_top = y_counts.index[0]
        cluster_indices.setdefault(y_top, []).append((i, y_counts, indices))

    sampled_points = []
    for y_top, xs in cluster_indices.items():
        print(y_top, len(xs))
        xs = sorted(xs, key=lambda x: (x[1]/x[1].sum()).iloc[0], reverse=True)
        for (i, c, indices) in xs[:n_clusters_per_class]:
            print(i, c)
            sampled_points.extend(indices)
        # print(i, len(indices), pd.Series(y[indices]).value_counts().iloc[:2].to_dict())
        # if len(indices) > n_pts_per_cluster:
        #     indices = np.random.choice(indices, n_pts_per_cluster, replace=False)
        # sampled_points.extend(indices)
    return np.array(sorted(sampled_points))


def dump_db(db_fname, df):
    db = lancedb.connect(db_fname)
    table_names = db.table_names()

    # ---------------------- Write out table for each object --------------------- #

    for object_name, odf in tqdm.tqdm(df.groupby('object')):
        if object_name != 'tortilla': continue
        if object_name in table_names:
            # if not overwrite:
            #     print("table", object_name, 'exists')
            #     return
            db.drop_table(object_name)
        print(object_name, len(odf))
        print(odf.describe())
        # idx = cluster_reduce(np.array(list(odf.vector.values)), odf.super_simple_state.values)
        # odf = odf.iloc[idx]
        # print(odf.describe())
        tbl = db.create_table(object_name, data=odf)#.iloc[idx]



import ipdb
@ipdb.iex
def build(config_name, overwrite=False):
    cfg = get_cfg(config_name)
    tree = pt.tree(cfg.DATASET.ROOT, {
        'embeddings1/{field_name}/{video_id}/{emb_type}/{track_id}.npz': 'emb_file',
        '{emb_type}.lancedb': 'db_fname',
    })
    # emb_dir = os.path.join(cfg.DATASET.ROOT, 'embeddings1', cfg.EVAL.DETECTION_NAME)
    # emb_types = cfg.EVAL.EMBEDDING_TYPES

    for emb_type in ['clip']: #emb_types:
        # ---------------------- Load the embeddings and states ---------------------- #

        data_file_pattern = tree.emb_file.specify(emb_type=emb_type).glob_format() #f'{emb_dir}/*/{emb_type}/*.npz'
        df = load_data(cfg, data_file_pattern)

        # ----------------------------- Open the database ---------------------------- #
        # db_fname = os.path.join(cfg.DATASET.ROOT, f'{cfg.EVAL.DETECTION_NAME}_{emb_type}.lancedb')
        db_fname = tree.db_fname.format(emb_type=emb_type)
        dump_db(db_fname, df)

def vis(db_fname):
    db = lancedb.connect(db_fname)
    for name in db.table_names():
        print(name)
        df = db[name].to_pandas()
        print(df.describe(include='all'))
    embed()

if __name__ == '__main__':
    import fire
    fire.Fire()