# LLFNet

## Dependencies
Please install the essential dependencies by running:
`pip install -r requirements.txt`

## Data pre-processing
Please organize the dataset in the following format:
```
Dataset/
├── train/
│   ├── case1.nii
│   ├── ...
│   ├── label/
│   │   ├── case1.nii
│   │   ├── ...
│
├── test/
│   ├── case1.nii
│   ├── ...
│   ├── label/
│   │   ├── case1.nii
│   │   ├── ...
```

## Training
Run `python train.py`

## Inference
Run `python inference.py`
