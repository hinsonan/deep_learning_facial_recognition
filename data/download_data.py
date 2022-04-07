import os
import argparse
import json

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--api_json_path', type=str, required=True)
# Parse the argument
args = parser.parse_args()

with open(args.api_json_path) as jsonfile:
    api_key = json.loads(jsonfile.read())

# env variables must be set before importing the kaggle module
os.environ['KAGGLE_USERNAME'] = api_key['username']
os.environ['KAGGLE_KEY'] = api_key['key']

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('sudarshanvaidya/random-images-for-face-emotion-recognition',path='greyscale_faces',unzip=True)
kaggle.api.dataset_download_files('amitprajapati191978/mood-detection',path='rgb_faces',unzip=True)
