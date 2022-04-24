# deep_learning_facial_recognition
Final Project for gatechs cs7643 deep learning class. Facial detection and emotional classification. 

## Dependencies 

```
pip install pytorch==1.10.2
pip install pillow
pip install opencv-python
pip install scikit-learn
pip install facenet_pytorch
```

## How to Run
There are two main parts for running the experiments and training the networks, The first part is gathering the metrics using the pre-trained facial detection models.

This is done by running the file `detector_pipeline.py`

`python detector_pipeline.py`

The output of this file will write the metrics to `experiment_results/detection_metrics`

The next part is training the emotional classifier

This is done in the file `train_pipeline.py`

`python train_pipeline.py`

This will take the config you have given in the main function as demonstrated below

```python
if __name__ == '__main__':
    pipeline = TrainingPipeline(f'config{os.sep}experiment1.yaml')
    pipeline.run_pipeline()
```
All configs files should go into the config folder. These Yaml files are how the training is driven. Please add new yaml files to run new experiments