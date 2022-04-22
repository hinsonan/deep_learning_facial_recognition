# How to download Kaggle datasets

1) `cd data`

2) `pip install kaggle`

3) Generate Kaggle API key (Go to kaggle and under your account there is a generate new API token)

4) save that json file somewhere on your computer

5) `python download_data.py --api_json_path <path_to_json>`

# How to download Detector dataset

1) Download from google drive https://drive.google.com/open?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&authuser=0

2) extract folder to data folder

3) Get the labels for the bounding boxes from http://shuoyang1213.me/WIDERFACE/ in the download section select the Face annotations and extract folder to data directory