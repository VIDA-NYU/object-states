


class DetectionAnnotator:
    def __init__(self):
        self.box_ann = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
        self.mask_ann = sv.MaskAnnotator()

    def annotate(self, frame, detections, labels):
        frame = self.mask_ann.annotate(frame, detections)
        frame = self.box_ann.annotate(frame, detections, labels=labels)
        return frame



def detectron2_to_sv(outputs, labels):
    detections = sv.Detections(
        xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
        mask=outputs["instances"].pred_masks.int().cpu().numpy(),
        confidence=outputs["instances"].scores.cpu().numpy(),
        class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
    )
    labels = labels[detections.class_id]
    return detections, labels

def egohos_to_sv(hoi_masks, hoi_class_ids, egohos):
    detections = sv.Detections(
        xyxy=masks_to_boxes(hoi_masks).cpu().numpy(),
        mask=hoi_masks.cpu().numpy(),
        class_id=hoi_class_ids,
    )
    labels = egohos.CLASSES[detections.class_id]
    return detections, labels

def tracks_to_sv(pred_mask, pred_boxes, track_ids, xmem, detic, egohos):
    # convert to Detection object for visualization
    track_detections = sv.Detections(
        mask=pred_mask.cpu().numpy(),
        xyxy=pred_boxes.cpu().numpy(),
        # class_id=np.array([xmem.tracks[i].label_count.most_common(1)[0][0] for i in track_ids]),
        # class_id=np.array([xmem.tracks[i].hoi_class_id for i in track_ids]),
        class_id=track_ids,
        tracker_id=track_ids,
    )

    # draw xmem detections
    labels = [
        [f'{detic.labels[l].split(",")[0]}' 
            for l, c in xmem.tracks[i].label_count.most_common(1)] + 
        [
            egohos.CLASSES[xmem.tracks[i].hoi_class_id],
            xmem.tracks[i].state_class_label,
        ]
        for i in track_detections.tracker_id
    ]
    labels = [' | '.join([l for l in ls if l]) for ls in labels]
    return track_detections, labels

