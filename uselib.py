import os
from ctypes import *

import cv2

HAS_GPU = True


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


lib = CDLL("libuse.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

if HAS_GPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]


class det_single_img():
    def __init__(self, configPath="cfg/vehicle.cfg", weightPath="cfg/vehicle_final.weights",
                 metaPath="cfg/vehicle.data", gpu_id=0):
        self.netMain, self.altNames = None, None
        if HAS_GPU:
            set_gpu(gpu_id)
        self.netMain = load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                result = match.group(1) if match else None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            self.altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
            with open(configPath) as fid:
                metaContents = fid.read()
                import re
                match = re.search("width *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                w = match.group(1) if match else None
                match = re.search("height *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                h = match.group(1) if match else None
                self.size = (int(float(h)), int(float(w)))

        except Exception:
            raise

        self._seconds = 0

    def detect_image(self, net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        letter_box = 0
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)

        num = pnum[0]
        print(num)
        if nms:
            do_nms_sort(dets, num, meta, nms)
        res = []
        for j in range(num):
            for i in range(meta):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        free_detections(dets, num)
        return res

    def detect(self, image_src, format="RGB"):
        try:
            try:
                if image_src.shape:
                    d_image = make_image(*self.size[:2], 3)
            except:
                image_src = cv2.imread(image_src)
                format = "BGR"
                d_image = make_image(*self.size[:2], 3)
            frame_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB) if format == "BGR" else image_src
            frame_resized = cv2.resize(frame_rgb, self.size, interpolation=cv2.INTER_LINEAR)
            copy_image_from_bytes(d_image, frame_resized.tobytes())
            detections = self.detect_image(self.netMain, len(self.altNames), d_image, thresh=0.25)
            return detections
        except:
            raise

    def getsize(self):
        return self.size

# import cv2
# from uselib import det_single_img
# yoyo=det_single_img()
# img=cv2.imread('lkjldkjjkkjshdkh2k3234hjk23.jpg')
# dd=yoyo.detect(img)

