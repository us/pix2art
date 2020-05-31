# Pix2Art
Our mission is to develop a tool for artists by using specific datasets (E.g.Rose, Daisy, Tulip, etc.). The purpose of this tool is to inspire the artists from a computer vision perspective.
## Tested Dataset
We trained pix2pix model with [Best Artworks of All Time](kaggle.com/ikarus777/best-artworks-of-all-time) dataset that is a collection of artworks of the 50 most influential artists of all time. The story behind this dataset is a challenge between a man and his girlfriend, they are challenging themselves like who is the best at guessing the artist behind an artwork. Then the man decides to use the power of machine learning to defeat his girlfriend and creates this dataset by scraping the internet.
## Usage
1. Download [frozen_model.pb](https://drive.google.com/file/d/1CjdINGYDAwMGWsSgxUYCyHOOBm5bkdIo/view?usp=sharing)
2. Clone repo.
3. pip install -r requirements.txt  # Install requirements.
4. python detect_edges_image.py   # Run model (default is running project with webcam)
## Create Dataset
python detect_edges_image.py --dataset_path=[path_of_dataset] --output_path=[dataset_output] 