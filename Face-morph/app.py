import os

from flask import Flask, render_template, redirect,request, flash, send_file
import faceMorph

from config import Config

app = Flask(__name__, static_url_path='')
app.config.from_object(Config)


@app.route('/<filename>', methods=['GET'])
def get_file(filename):
    """ Taking a string of filename as input,
        retrieving the file location though filename
        return a image file based on the location
    """
    return send_file(filename, as_attachment=True,
                     mimetype='video/x-matroska', last_modified=True)


@app.route('/<filename>', methods=['GET', 'POST'])
def download_file(filename):
    """ Taking a string of filename as input,
        retrieving the file location though filename
        return a image file based on the location
    """
    return send_file(filename, as_attachment=True)


@app.route('/')
def index():
    """ Setting up the starting page of webapp
    """
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    """ Taking two images file and a morphing rate
        as inputs through a POST request in index.html
        and uploading the images to the server.
        Then pass the url of two images and morphing rate
        to makeMorph function in faceMorph.py to produce
        a morphed image.
        If morph succeeds, return url of morphed result and
        both input images and the name of both input images.
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file1' not in request.files:
            flash('No file01 part')
            return redirect(request.url)

        if 'file2' not in request.files:
            flash('No file02 part')
            return redirect(request.url)
        file1 = request.files['file1']
        file2 = request.files['file2']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file1.filename == '':
            flash('No selected file01')
            return redirect(request.url)

        if file2.filename == '':
            flash('No selected file02')
            return redirect(request.url)

        fps = int(request.values['morph1'])

        if type(fps) != int or fps < 2 or fps > 20 :
            return render_template('index.html')

        # work on this to make it similar to our part
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = file1.filename
            filename2 = file2.filename
            save_to1 = (os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            save_to2 = (os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            file1.save(save_to1)
            file2.save(save_to2)

            morph_result = faceMorph.make_morph(save_to1, save_to2, fps)
            if type(morph_result) != str:
                flash("The model cannot learn one of the images points. Make sure you upload\
                a clear human face like the examples in instructions page")
                return render_template('index.html', morph="warning.jpg", filename="warning.jpg",\
                      f1_name=filename1.split(".")[0], f2_name=filename2.split(".")[0])
            return render_template('index.html', morph=morph_result,
                                   filename=morph_result, f1=filename1, f2=filename2,
                                   f1_name=filename1.split(".")[0],
                                   f2_name=filename2.split(".")[0])

    return render_template('index.html')


# allowed image types
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# is file allowed to be uploaded?


def allowed_file(filename):
    """Takes in a string as the input and check filename extension
        Return true if filename have correct extensions
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
