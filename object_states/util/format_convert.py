import cv2
import numpy as np
import torch
import fiftyone as fo
import supervision as sv
from xmem import XMem
from .util.video import XMemSink




# ---------------------------------------------------------------------------- #
#                           Converting to Supervision                          #
# ---------------------------------------------------------------------------- #


def fo_to_sv(fo_detections, shape):
    # get detections list
    fo_detections = getattr(fo_detections, 'detections', None) or []
    ds = [d for d in fo_detections if d.mask is not None]

    # convert to sv Detections object
    labels = np.array([d.label for d in ds], dtype=str)
    detections = sv.Detections(
        xyxy=np.array([xywhn2xyxy(d.bounding_box, shape) for d in ds]),
        track_ids=np.array([d.index for d in ds]),
        confidence=np.array([d.confidence for d in ds]),
        mask=np.array([d.to_segmentation(frame_size=shape, target=1).mask for d in ds]),
    )
    return detections, labels



# ---------------------------------------------------------------------------- #
#                            Converting to FiftyOne                            #
# ---------------------------------------------------------------------------- #



def detectron_to_fo(outputs, shape):
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    for xyxy, score, cls_id, mask in zip(
        instances.pred_boxes, # top-left, bottom-right xyxy
        instances.scores, # confidences
        instances.pred_classes, # class_ids
        instances.pred_masks  # segmentation masks (binary masks)
    ):
        x1, y1, x2, y2 = xyxy
        fo_mask = mask.numpy()[int(y1):int(y2), int(x1):int(x2)]
        detection = fo.Detection(
            label=detic.labels[cls_id], 
            confidence=float(score), 
            bounding_box=xyxy2xywhn(xyxy, shape), 
            mask=fo_mask)
        detections.append(detection)
    return fo.Detections(detections=detections)



# ---------------------------------------------------------------------------- #
#                               FiftyOne Helpers                               #
# ---------------------------------------------------------------------------- #



def detection2mask(d, orig_shape, new_shape):
    # the mask of a detection object is relative to the bounding box
    # convert to segmentation object
    return torch.as_tensor(cv2.resize(d.to_segmentation(frame_size=orig_shape[:2], target=1).mask, new_shape[:2]))

def mask2detection(mask):
    # the mask of a detection object needs to be relative to the bounding box
    # convert to segmentation object
    return fo.Segmentation(mask=mask.cpu().numpy()).to_detections().detections[0]




# ---------------------------------------------------------------------------- #
#                                  Coordinates                                 #
# ---------------------------------------------------------------------------- #



def xyxy2xywhn(xyxy, shape):
    xyxy[..., 2:] -= xyxy[..., :2]
    xyxy[..., 0] /= shape[1]
    xyxy[..., 1] /= shape[0]
    xyxy[..., 2] /= shape[1]
    xyxy[..., 3] /= shape[0]
    return xyxy

def xywhn2xyxy(xyxy, shape):
    xyxy[..., 2:] += xyxy[..., :2]
    xyxy[..., 0] *= shape[1]
    xyxy[..., 1] *= shape[0]
    xyxy[..., 2] *= shape[1]
    xyxy[..., 3] *= shape[0]
    return xyxy
