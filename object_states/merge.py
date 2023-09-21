




def delete_objects(video_sample, indices, field):
    video_path = video_sample.filepath
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    for i in tqdm.tqdm(range(1, video_info.total_frames+1)):
        finfo = video_sample.frames[i]
        detections = finfo[field].detections
        finfo[field] = fo.Detections(detections=[det for det in detections if det.index not in indices])
        finfo.save()
    return None

def merge_objects(video_sample, indices, field):
    video_path = video_sample.filepath
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)

    shape = (int(W/video_info.width*video_info.height), W)

    for i in tqdm.tqdm(range(1, video_info.total_frames+1)):
        finfo = video_sample.frames[i]
        detections = finfo[field].detections

        labels = [det.label for det in detections if det.index in indices]
        if not len(labels): continue
        confidences = [det.confidence for det in detections if det.index in indices]
        bounding_boxes = [det.bounding_box for det in detections if det.index in indices]
        bounding_boxes = torch.as_tensor(bounding_boxes, device=device)
        masks = [det.mask for det in detections if det.index in indices]

        bounding_boxes = xywh2xyxy(bounding_boxes, shape)
        masks = [fomask2fullmask(mask, bounding_box, shape) for mask, bounding_box in zip(masks, bounding_boxes)]

        box = torch.as_tensor([torch.min(bounding_boxes[:,0]), torch.min(bounding_boxes[:,1]),
            torch.max(bounding_boxes[:,2]), torch.max(bounding_boxes[:,3])])
        x1, y1, x2, y2 = np.round(box.tolist())
        box = xyxy2xywh(box, shape)

        mask = sum(masks).astype(bool)
        mask = mask[int(y1):int(y2), int(x1):int(x2)]

        detections = [det for det in detections if det.index not in indices]
        detections.append(fo.Detection(label=labels[0], confidence=confidences[0], bounding_box=box, mask=mask, index=min(indices)))

        finfo[field] = fo.Detections(detections=detections)
        finfo.save()
    return None