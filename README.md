# SLIC - Semantic Learning for Image Compression
---

*The server files are tested only on windows. There might be a few issues running on Linux/Mac because of varying file separator. (Most of the issues can be resolved by updating generate_map.py and combine_images.py)*

For reference, Kodak original and compressed images are kept in the code folder.

### Requirements

__Modules required to test__
To run the server on local machine, Python 3.6 or above is needed along with the following modules

* flask
* tensorflow
* numpy
* matplotlib
* pillow
* scikit-image
* pandas
* scipy
 
No other special modules are required for training. Version details for all the modules is available in requirements.txt file


### How to run

__GUI/Server__

* Make sure the trained model files are present in models/ folder
* Make sure the folders and files mentioned in the next section are present.
* Run using the following command
```
python3 server.py
```
* The server will be started on localhost:5000

__Training__
* Make sure you have the dataset downloaded in the data folder along with the pickle files
* Make sure the folders and files mentioned in the next section are present.
* Update params.py, if required
* Run using the following command
```
python3 train_resnet.py
```
* The training will start for 200 epochs by default, with learning rate as 0.001



### File structure
* models - contains the trained model file
* static - contains some static CSS, JS, image files
* templates - contains the HTML templates for the website
* combine_images.py - methods to encode using JPEG
* frameCapture.py - To test video compression by extracting frames
* generate_map.py - methods to generate heatmap and MS-ROI
* get_metrics.py - methods to calculate PSNR, SSIM
* params.py - params for tarining
* README.md - this file
* requirements.txt  
* resnet_model.py - model architecture file
* saveDataNp.py - to improve train performance
* server.py - flask server
* train_resnet.py - to train resnet
* util.py - utils funtions
