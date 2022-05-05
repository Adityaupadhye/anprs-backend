from os.path import splitext

import cv2
from keras.models import model_from_json, Sequential, load_model
import numpy as np

from classify import local_utils


# helper functions
def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


# class for license plate recognition
class LPR:

    wpod_net_path = 'models/wpod-net.json'

    def load_lpr_model(self):
        try:
            path = splitext(self.wpod_net_path)[0]
            with open('%s.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json, custom_objects={})
            model.load_weights('%s.h5' % path)
            print("LPR Model Loaded successfully...")
            print("Detecting License Plate ... ")
            return model
        except Exception as e:
            print(e)

    # wpod_net_model = load_lpr_model()

    # detect license plate
    def get_plate(self, image_path, Dmax=608, Dmin=608):

        vehicle_img = preprocess_image(image_path)
        ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        wpod_net_model = self.load_lpr_model()
        _, LpImg, _, cor = local_utils.detect_lp(wpod_net_model, vehicle_img, bound_dim, lp_threshold=0.5)
        return vehicle_img, LpImg, cor

    # save image
    def save_predicted_img(self, img_path):
        try:
            vehicle, LpImg, cor = self.get_plate(img_path)
            img = cv2.convertScaleAbs(LpImg[0], alpha=255.0)
            cv2.imwrite('results/res.jpg', img)
            return True
        except Exception as e:
            print(e)
            return False

    # wrapper for performing lpr
    def perform_lpr(self, original_img_path):
        print('performing lpr...')
        res = self.save_predicted_img(original_img_path)
        print('LPR results saved')
        return res


# class for optical character recognition
class OCR:

    ocr_model_path = 'models/ocr_model.h5'

    # lp_image_path = 'results/res.jpg'

    def find_contours(self, dimensions, img):

        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]

        img_ht, img_wt = np.shape(img)

        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

        ii = cv2.imread('results/contour.jpg')

        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs:
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

            # if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            if 0.40*img_wt>intWidth>0.01*img_wt and 0.75*img_ht>intHeight>0.40*img_ht:
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

    def segment_characters(self, image):

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

        if not cv2.imwrite('results/contour.jpg', img_binary_lp):
            raise Exception('Could not write contours image')

        print('image segmented...')

        char_list = self.find_contours(dimensions, img_binary_lp)

        print('contours found...')

        return char_list

    # lp_image = cv2.imread(lp_image_path)

    model: Sequential = load_model('models/ocr_model.h5')

    def fix_dimension(self, img):
        new_img = np.zeros((28, 28, 3))
        for i in range(3):
            new_img[:, :, i] = img
            return new_img

    def get_results(self):
        print('started ocr...')

        lp_image_path = 'results/res.jpg'
        lp_image = cv2.imread(lp_image_path)
        # lp_canny_image = cv2.Canny(lp_image)
        # lp_canny_image = cv2.resize(lp_canny_image, (32,32))


        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, c in enumerate(characters):
            dic[i] = c

        output = []
        segmented_chars = self.segment_characters(lp_image)
        for i, ch in enumerate(segmented_chars):
            print(type(ch), ch.shape)
            # img_canny = cv2.Canny(np.uint8(ch), 32, 32)
            img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
            img = self.fix_dimension(img_)
            img = img.reshape((1, 28, 28, 3))
            print(img.shape)
            # img = img.reshape((1, 28, 28, 3))

            # im = np.array(img)
            # im = img/255.0
            y_ = self.model.predict(img)

            idx = np.argmax(y_)

            character = dic[idx]
            output.append(character)

        plate_number = ''.join(output)

        print('ocr done.')

        return plate_number


# wrapper for whole process
class Recognizer:
    name = ''
    lpr_model_path = ''


lpr = LPR()
ocr = OCR()
