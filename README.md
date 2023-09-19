# Object State Classification


## Step 1: Extract Detic+XMem
script: `predict.py data_dir dataset`

Processing:
 - Detic
 - EgoHOS to prioritize tracks??
 - XMem

Output:
 - FiftyOne Dataset
 - Rendered video
 - Rendered track videos

## Manual Step: Validate / Merge for somewhat trustworthy ground truth
script: `merge.py dataset video_id 1+3 -4`

Expects:
 - FiftyOne Dataset

Processing:
 - merge/delete tracks

Output:
 - Validated FiftyOne Dataset

## Step 2: Convert FiftyOne dataset to embeddings with augmentations
script: `embed.py dataset`

Processing:
 - augmentation
    - rotation
    - image augmentation
 - embeddings:
    - CLIP
    - detic stages
    - detic stages averaged
    - detic raw

Output:
 - Sample of augmented frames (by label)
 - embedding npz labeled with class and step

## Final: Evaluate
script: `eval.py dataset`

Expects:
 - embedding npz labeled with class and step
  - ~>=1 hand validated
  - hand label tortilla in packaging
  - others automatic

Processing:
 - given train-test split get validation accuracy for each critical object
    - load training set
    - train nearest neighbors
    - for each critical object
      - classify step
      - compare step to ground truth step
      - MSE to step percentage
      - plot step+MSE over time
      - count separate polygons over time
