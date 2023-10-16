# import ray
# from ray import serve
# from ray.serve.handle import DeploymentHandle
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
import lancedb

import clip
from detic import Detic
from detic.inference import load_classifier
from egohos import EgoHos
from xmem import XMem

from detectron2.structures import Boxes, Instances, pairwise_iou
from torchvision.ops import masks_to_boxes

from ..util.nms import asymmetric_nms, mask_iou
from ..util.vocab import prepare_vocab
from .download import ensure_db

from IPython import embed

log = logging.getLogger(__name__)

# ray.init()


class CustomTrack(XMem.Track):
    hoi_class_id = 0
    state_class_label = ''
    def __init__(self, track_id, t_obs, n_init=3, **kw):
        super().__init__(track_id, t_obs, n_init, **kw)
        self.label_count = Counter()
        self.z_clips = {}

# IGNORE_CLASSES = ['table', 'dining_table', 'table-tennis_table', 'person']

class ObjectDetector:
    def __init__(self, vocabulary, state_db_fname=None, xmem_config={}, conf_threshold=0.5, device='cuda'):
        # initialize models
        self.device = device
        self.detic = Detic([], masks=True, one_class_per_proposal=3, conf_threshold=conf_threshold).eval()
        self.conf_threshold = conf_threshold

        self.egohos = EgoHos('obj1').eval()
        self.egohos_type = np.array(['', 'hand', 'hand', 'obj', 'obj', 'obj', 'obj', 'obj', 'obj', 'cb'])
        self.egohos_hand_side = np.array(['', 'left', 'right', 'left', 'right', 'both', 'left', 'right', 'both', ''])

        self.xmem = XMem({
            'top_k': 30,
            'mem_every': 30,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 6,
            'max_mid_term_frames': 12,
            'max_long_term_elements': 1000,
            'tentative_frames': 3,
            'tentative_age': 3,
            'max_age': 60,  # in steps
            # 'min_iou': 0.3,
            **xmem_config,
        }, Track=CustomTrack).cuda().eval()

        # load vocabularies
        if vocabulary.get('base'):
            _, open_meta, _ = load_classifier(vocabulary['base'], prepare=False)
            base_prompts = open_meta.thing_classes
        else:
            base_prompts = []
        tracked_prompts, tracked_vocab = prepare_vocab(vocabulary['tracked'])
        untracked_prompts, untracked_vocab = prepare_vocab(vocabulary.get('untracked') or [])

        # get base prompts
        remove_vocab = set(vocabulary.get('remove') or ()) | set(tracked_prompts) | set(untracked_prompts)
        base_prompts = [c for c in base_prompts if c not in remove_vocab]

        # get base vocab
        equival_map = vocabulary.get('equivalencies') or {}
        base_vocab = [equival_map.get(c, c) for c in base_prompts]

        # combine and get final vocab list
        full_vocab = list(tracked_vocab) + list(untracked_vocab) + base_vocab
        full_prompts = list(tracked_prompts) + list(untracked_prompts) + base_prompts

        self.tracked_vocabulary = np.asarray(list(set(tracked_vocab)))
        self.ignored_vocabulary = np.asarray(['IGNORE'])

        self.skill_clsf, _, _ = load_classifier(full_prompts, metadata_name='lvis+')
        self.skill_labels = np.asarray(full_vocab)
        self.skill_labels_is_tracked = np.isin(self.skill_labels, self.tracked_vocabulary)

        self.state_db_key = 'super_simple_state'
        self.obj_label_names = []
        if state_db_fname:
            state_db_fname = ensure_db(state_db_fname)
            self.obj_state_db = lancedb.connect(state_db_fname)
            self.obj_label_names = self.obj_state_db.table_names()
            self.obj_state_tables = {
                k: self.obj_state_db[k]
                for k in self.obj_label_names
            }
            print(f"State DB: {self.obj_state_db}")
            print(f'Objects: {self.obj_label_names}')
            # for name in self.obj_label_names:
            #     tbl.create_index(num_partitions=256, num_sub_vectors=96)
        # image encoder
        self.clip, self.clip_pre = clip.load("ViT-B/32", device=self.device)



    def predict_objects(self, image):
        # ----------------------------- Object Detection ----------------------------- #

        # predict objects
        detic_query = self.detic.build_query(image)
        outputs = detic_query.detect(self.skill_clsf, conf_threshold=self.conf_threshold, labels=self.skill_labels)
        instances = outputs['instances']
        instances = self._filter_detections(instances)
        return instances, detic_query
    
    def _filter_detections(self, instances):
        instances = instances[~np.isin(instances.pred_labels, self.ignored_vocabulary)]
        # filter out objects completely inside another object
        obj_priority = torch.from_numpy(np.isin(instances.pred_labels, self.tracked_vocabulary)).int()
        filtered, overlap = asymmetric_nms(instances.pred_boxes.tensor, instances.scores, obj_priority, iou_threshold=0.85)
        filtered_instances = instances[filtered.cpu().numpy()]
        # if Counter(instances.pred_labels.tolist()).get('tortilla', 0) > 1:
        #     embed()
        for i, i_ov in enumerate(overlap):
            if not len(i_ov): continue
            # get overlapping instances
            overlap_insts = instances[i_ov.cpu().numpy()]
            log.info(f"object {filtered_instances.pred_labels[i]} filtered {overlap_insts.pred_labels}")

            # merge overlapping detections with the same label
            overlap_insts = overlap_insts[overlap_insts.pred_labels == filtered_instances.pred_labels[i]]
            if len(overlap_insts):
                log.info(f"object {filtered_instances.pred_labels[i]} merged {len(overlap_insts)}")
                filtered_instances.pred_masks[i] |= torch.maximum(
                    filtered_instances.pred_masks[i], 
                    overlap_insts.pred_masks.max(0).values)
        return filtered_instances

    def predict_hoi(self, image):
        # -------------------------- Hand-Object Interaction ------------------------- #

        # predict HOI
        hoi_masks, hoi_class_ids = self.egohos(image)
        keep = hoi_masks.sum(1).sum(1) > 4
        hoi_masks = hoi_masks[keep]
        hoi_class_ids = hoi_class_ids[keep.cpu().numpy()]
        # create detectron2 instances
        instances = Instances(
            image.shape,
            pred_masks=hoi_masks,
            pred_boxes=Boxes(masks_to_boxes(hoi_masks)),
            pred_hoi_classes=hoi_class_ids)
        # get a mask of the hands
        hand_mask = hoi_masks[self.egohos_type[hoi_class_ids] == 'hand'].sum(0)
        return instances, hand_mask

    def merge_hoi(self, other_detections, hoi_detections, detic_query):
        is_obj_type = self.egohos_type[hoi_detections.pred_hoi_classes] == 'obj'
        hoi_obj_detections = hoi_detections[is_obj_type]
        hoi_obj_masks = hoi_obj_detections.pred_masks
        hoi_obj_boxes = hoi_obj_detections.pred_boxes.tensor
        hoi_obj_hand_side = self.egohos_hand_side[hoi_detections.pred_hoi_classes[is_obj_type]]


        # ----------------- Compare & Merge HOI with Object Detector ----------------- #

        # get mask iou
        other_detections = [d for d in other_detections if d is not None]
        mask_list = [d.pred_masks for d in other_detections]
        det_masks = torch.cat(mask_list) if mask_list else torch.zeros(0, hoi_obj_masks.shape[1:])
        iou = mask_iou(det_masks, hoi_obj_masks)
        # add hand side interaction to tracks
        i = 0
        for d, b in zip(other_detections, mask_list):
            d.left_hand_interaction = iou[i:i+len(b), hoi_obj_hand_side == 'left'].sum(1)
            d.right_hand_interaction = iou[i:i+len(b), hoi_obj_hand_side == 'right'].sum(1)
            d.both_hand_interaction = iou[i:i+len(b), hoi_obj_hand_side == 'both'].sum(1)
            i += len(b)

        # ---------------------- Predict class for unlabeled HOI --------------------- #

        # get hoi objects with poor overlap
        hoi_is_its_own_obj = iou.sum(0) < 0.3
        # get labels for HOIs
        hoi_outputs = detic_query.predict(hoi_obj_boxes[hoi_is_its_own_obj], self.skill_clsf, labels=self.skill_labels)
        hoi_detections2 = hoi_outputs['instances']
        hoi_detections2.pred_masks = hoi_obj_detections.pred_masks[hoi_is_its_own_obj]
        hoi_is_its_own_obj = hoi_is_its_own_obj.cpu()
        hoi_detections2.left_hand_interaction = torch.as_tensor(hoi_obj_hand_side == 'left')[hoi_is_its_own_obj]
        hoi_detections2.right_hand_interaction = torch.as_tensor(hoi_obj_hand_side == 'right')[hoi_is_its_own_obj]
        hoi_detections2.both_hand_interaction = torch.as_tensor(hoi_obj_hand_side == 'both')[hoi_is_its_own_obj]
        # TODO: add top K classes and scores
        return hoi_detections2

    def filter_objects(self, detections):
        return detections, detections

    def track_objects(self, image, detections, negative_mask=None):
        # 
        det_mask = None
        # other_mask = None
        frame_detections = None
        tracked_detections = None
        # is_tracked = None
        # mask_labels = None
        det_scores = None
        if detections is not None:
            # mask_labels = detections.pred_labels
            # is_tracked = np.isin(detections.pred_labels, self.tracked_vocabulary)
            # tracked_detections = detections[is_tracked]
            # frame_detections = detections[~is_tracked]
            # other_mask = frame_detections.pred_masks
            det_scores = detections.pred_scores
            det_mask = detections.pred_masks
            frame_detections = detections

        # run xmem
        pred_mask, track_ids, input_track_ids = self.xmem(
            image, det_mask, 
            negative_mask=negative_mask, 
            mask_scores=det_scores,
            tracked_labels=self.skill_labels_is_tracked,
            only_confirmed=True
        )
        # update label counts
        tracks = self.xmem.tracks
        if input_track_ids is not None and detections is not None:
            for (ti, label) in zip(input_track_ids, detections.pred_labels):
                if ti >= 0:
                    tracks[ti].label_count.update([label])

        instances = Instances(
            image.shape,
            pred_boxes=Boxes(masks_to_boxes(pred_mask)),
            pred_masks=pred_mask,
            pred_labels=np.array([tracks[i].label_count.most_common(1)[0][0] for i in track_ids]),
            track_ids=torch.as_tensor(track_ids),
        )
        return instances, frame_detections

    def predict_state(self, image, detections):
        states = []
        has_state = np.isin(detections.pred_labels, self.obj_label_names)
        dets = detections[has_state]
        i_z = {k: i for i, k in enumerate(np.where(has_state)[0])}
        Z_imgs = self._encode_boxes(image, dets.pred_boxes.tensor) if len(dets) else None
        for i in range(len(detections)):
            pred_label = detections.pred_labels[i]
            if has_state[i]:
                df = self.obj_state_tables[pred_label].search(Z_imgs[i_z[i]].cpu().numpy()).limit(11).to_df()
                state = df[self.state_db_key].value_counts()
                state = state / state.sum()
            else:
                state = pd.Series()
            states.append(state.to_dict())
        # detections.__dict__['pred_states'] = states
        detections.pred_states = np.array(states)
        return detections

    def _encode_boxes(self, img, boxes):
        # BGR
        # encode each bounding box crop with clip
        # print(f"Clip encoding: {img.shape} {boxes.shape}")
        # for x, y, x2, y2 in boxes.cpu():
        #     Image.fromarray(img[
        #         int(y):max(int(np.ceil(y2)), int(y+2)),
        #         int(x):max(int(np.ceil(x2)), int(x+2)),
        #         ::-1]).save("box.png")
        #     input()
        Z = self.clip.encode_image(torch.stack([
            self.clip_pre(Image.fromarray(img[
                int(y):max(int(np.ceil(y2)), int(y+2)),
                int(x):max(int(np.ceil(x2)), int(x+2)),
                ::-1]))
            for x, y, x2, y2 in boxes.cpu()
        ]).to(self.device))
        Z /= Z.norm(dim=1, keepdim=True)
        return Z
    
    def classify(self, Z, labels):
        outputs = []
        for z, l in zip(Z, labels):
            z_cls, txt_cls = self.classifiers[l]
            out = (z @ z_cls.t()).softmax(dim=-1).cpu().numpy()
            i = np.argmax(out)
            outputs.append(txt_cls[i])
        return np.atleast_1d(np.array(outputs))

    def forward(self, img, boxes, labels):
        valid = self.can_classify(labels)
        if not valid.any():
            return np.array([None]*len(boxes))
        labels = np.asanyarray(labels)
        Z = self.encode_boxes(img, boxes[valid])
        clses = self.classify(Z, labels[valid])
        all_clses = np.array([None]*len(boxes))
        all_clses[valid] = clses
        return all_clses


class Perception:
    def __init__(self, *a, detect_every_n_seconds=0.5, **kw):
        self.detector = ObjectDetector(*a, **kw)
        self.detect_every_n_seconds = detect_every_n_seconds
        self.detection_timestamp = -detect_every_n_seconds

    @torch.no_grad()
    def predict(self, image, timestamp):
        # ---------------------------------------------------------------------------- #
        #                           Detection: every N frames                          #
        # ---------------------------------------------------------------------------- #

        detections = detic_query = hoi_detections = hand_mask = None
        is_detection_frame = (timestamp - self.detection_timestamp) >= self.detect_every_n_seconds
        if is_detection_frame:
            self.detection_timestamp = timestamp

            # -------------------------- First we detect objects ------------------------- #
            # Detic: 

            detections, detic_query = self.detector.predict_objects(image)

            # ------------------ Then we detect hand object interactions ----------------- #
            # EgoHOS:

            hoi_detections, hand_mask = self.detector.predict_hoi(image)

        # ---------------------------------------------------------------------------- #
        #                             Tracking: Every frame                            #
        # ---------------------------------------------------------------------------- #

        # ------------------------- Then we track the objects ------------------------ #
        # XMem:

        track_detections, frame_detections = self.detector.track_objects(image, detections, negative_mask=hand_mask)

        # ---------------------------------------------------------------------------- #
        #                            Predicting Object State                           #
        # ---------------------------------------------------------------------------- #

        # -------- For objects with labels we care about, classify their state ------- #
        # LanceDB:

        # predict state for tracked objects
        track_detections = self.detector.predict_state(image, track_detections)
        # predict state for untracked objects
        # if frame_detections is not None:
        #     frame_detections = self.detector.predict_state(image, frame_detections)


        # ----- Merge our multi-model detections into a single set of detections ----- #
        # IoU between tracks+frames & hoi:

        if is_detection_frame:
            # Merging HOI into track_detections, frame_detections, hoi_detections
            hoi_detections = self.detector.merge_hoi(
                [track_detections, frame_detections],
                hoi_detections,
                detic_query)

        self.timestamp = timestamp
        return track_detections, frame_detections, hoi_detections


    def serialize_detections(self, detections):
        bboxes = detections.pred_boxes.tensor.cpu().numpy()
        labels = detections.pred_labels
        track_ids = detections.track_ids.cpu().numpy() if detections.has('track_ids') else None

        hand_object = { k: f'{k}_hand_interaction' for k in ['left', 'right', 'both'] }
        hand_object = {
            k: detections.get(kk).cpu().numpy()
            for k, kk in hand_object.items() 
            if detections.has(kk)}

        possible_labels = None
        if detections.has('topk_scores'):
            possible_labels = [
                {k: v for k, v in zip(ls.tolist(), ss.tolist()) if v > 0}
                for ls, ss in zip(detections.topk_labels, detections.topk_scores.cpu().numpy())
            ]

        states = detections.pred_states if detections.has('pred_states') else None

        output = []
        for i in range(len(detections)):
            data = {
                'xyxyn': bboxes[i],
                'label': labels[i],
            }
            if hand_object:
                ho = {k: x[i] for k, x in hand_object.items()}
                data['hand_object'] = {
                    **ho,
                    'any': max(ho.values(), default=0),
                }
            if possible_labels:
                data['possible_labels'] = possible_labels[i]

            if states is not None:
                data['state'] = states[i]

            if track_ids is not None:
                data['segment_track_id'] = track_ids[i]
            output.append(data)
        return output
