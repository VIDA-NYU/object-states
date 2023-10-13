import torch
import numpy as np
# from IPython import embed


def mask_iou(a, b, eps=1e-7):
    a, b = a[:, None], b[None]
    overlap = (a * b) > 0
    union = (a + b) > 0
    return 1. * overlap.sum((2, 3)) / (union.sum((2, 3)) + eps)


def asymmetric_nms(boxes, scores, priority=None, iou_threshold=0.99):
    # # Get indices that would sort the tensor along the first column
    # sorted_indices = torch.argsort(x[:, 0])

    # # Use these indices to rearrange the rows of the tensor
    # sorted_x = x[sorted_indices]

    # # If values in the first column are equal, sort along the second column
    # sorted_indices_second_col = torch.argsort(sorted_x[:, 1])

    # # Use these indices to rearrange the rows again
    # final_sorted_x = sorted_x[sorted_indices_second_col]

    # nn=len(boxes)
    # Sort boxes by their confidence scores in descending order
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # indices = torch.argsort(area, descending=True)
    # if priority is not None:
    #     indices = torch.argsort(priority, descending=True)
    # indices = np.argsort(scores)[::-1]
    if priority is not None:
        indices = torch.as_tensor(np.lexsort((
            -area.cpu().numpy(),
            -priority.cpu().numpy(), 
        )), device=area.device)
    else:
        indices = torch.argsort(area, descending=True)
    boxes = boxes[indices]
    scores = scores[indices]

    selected_indices = []
    overlap_indices = []
    while len(boxes) > 0:
        # Pick the box with the highest confidence score
        b = boxes[0]
        selected_indices.append(indices[0])

        # Calculate IoU between the picked box and the remaining boxes
        zero = torch.tensor([0], device=boxes.device)
        intersection_area = (
            torch.maximum(zero, torch.minimum(b[2], boxes[1:, 2]) - torch.maximum(b[0], boxes[1:, 0])) * 
            torch.maximum(zero, torch.minimum(b[3], boxes[1:, 3]) - torch.maximum(b[1], boxes[1:, 1]))
        )
        smaller_box_area = torch.minimum(area[0], area[1:])
        iou = intersection_area / (smaller_box_area + 1e-7)

        # Filter out boxes with IoU above the threshold

        overlap_indices.append(indices[torch.where(iou > iou_threshold)[0] + 1])
        filtered_indices = torch.where(iou <= iou_threshold)[0]
        indices = indices[filtered_indices + 1]
        boxes = boxes[filtered_indices + 1]
        scores = scores[filtered_indices + 1]
        area = area[filtered_indices + 1]

    selected_indices = (
        torch.stack(selected_indices) if selected_indices else 
        torch.zeros([0], dtype=torch.int32, device=boxes.device))
    # print(nn, overlap_indices)
    # if nn>1 and input():embed()
    return selected_indices, overlap_indices
