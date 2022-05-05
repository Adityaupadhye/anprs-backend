import base64
import json

import tensorflow as tf
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from db.models import Vehicle_owners, Owners
from ml.anprs import lpr, ocr

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
from . import local_utils
# from local_utils import detect_lp
from os.path import splitext
from tensorflow.keras.models import model_from_json

import keras
from keras.models import Sequential


def index(request):
    if request.method == "POST":

        f = request.FILES['sentFile']
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        print(file_url)

        def load_model(path):
            try:
                path = splitext(path)[0]
                with open('%s.json' % path, 'r') as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json, custom_objects={})
                model.load_weights('%s.h5' % path)
                print("Model Loaded successfully...")
                print("Detecting License Plate ... ")
                return model
            except Exception as e:
                print(e)

        wpod_net_path = "models/wpod-net.json"
        wpod_net = load_model(wpod_net_path)

        def preprocess_image(image_path, resize=False):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            if resize:
                img = cv2.resize(img, (224, 224))
            return img

        def get_plate(image_path, Dmax=608, Dmin=608):
            vehicle = preprocess_image(image_path)
            ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
            side = int(ratio * Dmin)
            bound_dim = min(side, Dmax)
            _, LpImg, _, cor = local_utils.detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
            return vehicle, LpImg, cor

        test_image_path = "media/pic.jpg"
        vehicle, LpImg, cor = get_plate(test_image_path)
        img = cv2.convertScaleAbs(LpImg[0], alpha=255.0)
        cv2.imwrite('results/res.jpg', img)

        def find_contours(dimensions, img):

            cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            lower_width = dimensions[0]
            upper_width = dimensions[1]
            lower_height = dimensions[2]
            upper_height = dimensions[3]

            cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

            ii = cv2.imread('results/contour.jpg')

            x_cntr_list = []
            target_contours = []
            img_res = []
            for cntr in cntrs:
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

                if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
                    x_cntr_list.append(intX)

                    char_copy = np.zeros((44, 24))
                    char = img[intY:intY + intHeight, intX:intX + intWidth]
                    char = cv2.resize(char, (20, 40))

                    cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)

                    char = cv2.subtract(255, char)

                    char_copy[2:42, 2:22] = char
                    char_copy[0:2, :] = 0
                    char_copy[:, 0:2] = 0
                    char_copy[42:44, :] = 0
                    char_copy[:, 22:24] = 0

                    img_res.append(char_copy)
                    # plt.show()

            indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
            img_res_copy = []
            for idx in indices:
                img_res_copy.append(img_res[idx])
            img_res = np.array(img_res_copy)

            return img_res

        def segment_characters(image):

            img_lp = cv2.resize(image, (333, 75))
            img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
            _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
            img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

            LP_WIDTH = img_binary_lp.shape[0]
            LP_HEIGHT = img_binary_lp.shape[1]

            img_binary_lp[0:3, :] = 255
            img_binary_lp[:, 0:3] = 255
            img_binary_lp[72:75, :] = 255
            img_binary_lp[:, 330:333] = 255

            dimensions = [LP_WIDTH / 6,
                          LP_WIDTH / 2,
                          LP_HEIGHT / 10,
                          2 * LP_HEIGHT / 3]

            cv2.imwrite('results/contour.jpg', img_binary_lp)

            char_list = find_contours(dimensions, img_binary_lp)

            return char_list

        plate = cv2.imread("results/res.jpg")
        char = segment_characters(plate)

        model: Sequential = keras.models.load_model('models/lpr_model.h5')

        def fix_dimension(img):
            new_img = np.zeros((28, 28, 3))
            for i in range(3):
                new_img[:, :, i] = img
                return new_img

        def show_results():
            dic = {}
            characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for i, c in enumerate(characters):
                dic[i] = c

            output = []
            for i, ch in enumerate(char):
                img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
                img = fix_dimension(img_)
                img = img.reshape(1, 28, 28, 3)
                y_ = model.predict(img)

                print('outputs=', y_[0], np.shape(y_))

                idx = np.argmax(y_[0])

                print(idx)

                character = dic[idx]
                output.append(character)
            plate_number = ''.join(output)

            return plate_number

        print(show_results())
        str = ""
        str = show_results()

        # obj = LP.objects.filter(veh_num=str)
        # # print(obj)
        # bool_val = False
        # if obj:
        #     bool_val = True
        #
        # print(bool_val)
        context = {"str": str}

        # response['name'] = str
        return render(request, 'homepage.html', context)
    else:
        return render(request, 'homepage.html')


@csrf_exempt
def detect(request):
    if request.method == 'POST':

        data = json.loads(request.body.decode('utf-8'))
        encoded_img = data['image']
        file_url = save_base64_image(encoded_img)
        print('saved file=', file_url)

        lpr_res = lpr.perform_lpr('.' + file_url)

        license_plate_number = '0'

        if lpr_res:
            license_plate_number = ocr.get_results()

        else:
            return JsonResponse(
                {
                    'status': False,
                    'message': 'Error in detecting plate',
                    'lpn': None
                }
            )

        return JsonResponse(
            {
                'status': True,
                'message': 'License Plate Detected',
                'lpn': license_plate_number
            }
        )

    elif request.method == 'GET':
        return render(request, 'homepage.html')


def save_base64_image(data: str):
    # _type, img_str = data.split(';base64,')
    # _name, ext = _type.split('/')
    data = ContentFile(base64.b64decode(data))
    saved_file_name = default_storage.save('photo.jpg', data)
    return default_storage.url(saved_file_name)


def test(request):
    return JsonResponse({
        'status': True,
        'message': 'Hello ANPRS!'
    })


def getOwnerDetails(request, lpn):

    if request.method == 'GET':

        fetched_ownerid = Vehicle_owners.objects.filter(plate_number=lpn)

        if not fetched_ownerid:
            return JsonResponse({
                'status': False,
                'message': 'Owner not found',
                'owner': None,
                'mobile': None,
                'mail': None
            })

        owner = Owners.objects.filter(id = fetched_ownerid[0].owner_id)
        print(lpn, owner[0].name)

        return JsonResponse({
            'status': True,
            'message': 'Owner found!',
            'owner': owner[0].name,
            'mobile': owner[0].mobile,
            'mail': owner[0].mail,
            'dept': owner[0].dept,
            'role': owner[0].role
        })

    return JsonResponse({
        'status': False
    })
