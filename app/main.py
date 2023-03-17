import base64
# from crypt import methods
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import app.utils.utils as utils
import app.utils.config as config
import app.utils.models as models


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/')
def index():
    return 'This is a test service'


@app.route('/leaf-chars', methods=['GET', 'POST'])
def upload_leaf_file():
    up_file = request.files['file']
    read_file = up_file.read()
    image = Image.open(BytesIO(read_file))
    w, h = image.size
    image = image.resize(config.cfg.phenot.img_size)    
    length_factor = ((w+h)/2) * config.cfg.phenot.f1
    area_factor = w * h * config.cfg.phenot.f2
    data1, data2, blade_pixels, vein_pixels = utils.draw_veins(image, resize_factor=length_factor)
    semantic_img_bytes_1 = base64.b64encode(data1)
    semantic_img_bytes_2 = base64.b64encode(data2)
    return jsonify({'image': f'{semantic_img_bytes_1}', 'veins': f'{semantic_img_bytes_2}', 'blade_area': f'{round(blade_pixels*area_factor, 2)} cm^2', 'vein_area': f'{round(vein_pixels*area_factor, 2)} cm^2'})


if __name__ == "__main__":
    # Only for debugging while developing
    # app.run(host='0.0.0.0', debug=True, port=80)
    app.run(debug=True)
