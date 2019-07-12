import os, time
from flask import Flask, render_template, request
from generate_map import compressImage
from combine_images import combineImage

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
	# The main home page rendering method.
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def upload_file():
    """
    Uploads the image, compresses it and renders the compressed image
    
    Returns
    -------
    HTML Page
        Page with compressed image
    """
    startTime = time.time()
    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Generate heatmap and MS-ROI
    hmPath, roiPath = compressImage(filename)

    print ("heatmap path ---> ", hmPath)
    print ("roi path ---> ", roiPath)

    # Compress image based on MS-ROI
    outPath, newSize, uncSize, psnr, ssim = combineImage (filename, roiPath)

    oriSize = os.path.getsize(filename)

    uis = str(round((uncSize/1024),5))+" KB ("+str(uncSize)+" bytes)"
    ois = str(round((oriSize/1024),5))+" KB ("+str(oriSize)+" bytes)"
    cis = str(round((newSize/1024),5))+" KB ("+str(newSize)+" bytes)"
    cr = round(float(uncSize)/float(newSize), 4)


    tt = str(round(time.time() - startTime, 2)) + " seconds"

    # Format dictionary to show the metrics
    propsDict = {"Uncompressed Size": uis, "Original Size": ois, "Compressed Size": cis, 
    "Compression Ratio": cr,"PSNR": round(psnr, 4), "SSIM": round(ssim, 4), "Time Taken": tt}

    print (propsDict)
    os.remove(filename)

    return render_template('index.html', compressed=True, outPath=outPath, propsDict=propsDict)

if __name__ == "__main__":
	app.run()
	app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
