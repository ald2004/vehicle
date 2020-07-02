import inspect
import json
import logging
import traceback

import cv2
import numpy as np
from flask import Flask, request, Response
from skimage.measure import compare_ssim

from utils import (
    setup_logger,
    base64toImageArray,
    det_single_img,
    kill_duplicate_by_score,
    convert_back_xyxy,
    convertBack,
    Instances,
    if_between_twoline,
)

frame = inspect.currentframe()
thing_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'trains']
vehicle_device_infos = {}
logger = setup_logger(log_level=logging.DEBUG)
app = Flask(__name__)
yoyo = det_single_img(configPath="cfg/vehicle.cfg", weightPath="cfg/vehicle_last.weights",
                      metaPath="cfg/vehicle.data", gpu_id=0)
app.run(debug=True, port=5123, host='0.0.0.0')


def deal_detect(deviceInfo, ssim_thres=.5):
    last_objs = deviceInfo.get('l_objs', None)
    if not last_objs: return None
    ret_info = {}
    ret_list = []

    try:
        c_image = deviceInfo.get('c_img')
        if not c_image: return None
        c_objs = []
        predicts, _ = yoyo.darkdetect()
        predicts = kill_duplicate_by_score(predicts, xou_thres=.85)
        for dd in predicts:
            # ('truck', 0.9885375499725342, (396.35833740234375, 338.2737121582031, 280.8970642089844, 172.4924774169922))
            # cc = [int(float(x)) for x in convert_back_xyxy(*yoyo.getsize(), *c_image.shape[:2], *convertBack(dd[2]))]
            xmin, ymin, xmax, ymax = [int(float(x)) for x in
                                      convert_back_xyxy(*yoyo.getsize(), *c_image.shape[:2], *convertBack(dd[2]))]
            centroid = Instances(**{"x": round((xmin + xmax) / 2, 6), "y": round((ymin + ymax) / 2, 6)})
            if if_between_twoline(deviceInfo['LINE_A'], deviceInfo['LINE_B'], centroid):
                c_objs.append(c_image[ymin:ymax, xmin:xmax, :])
        deviceInfo['l_objs'] = c_objs
        for lo in last_objs:
            for idx, co in enumerate(c_objs):
                # to be implemented first compute iou,and now compute ssim directly.
                imageA, imageB = lo, co
                if np.min(np.array(imageA.shape[:2]) - np.array(imageB.shape[:2])):
                    imageB = cv2.resize(imageB, imageA.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
                print(f"the 2 of pics shape, imageA:{imageA.shape},imageB:{imageB.shape}")
                score, _ = compare_ssim(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY), full=True)
                if score > ssim_thres:
                    ret_info['DEVICE_SN'] = deviceInfo.get('DEVICE_SN', None)
                    ret_info['THINGS_NAME'] = predicts[idx][0]
                    a, b, c, d = [int(float(x)) for x in convert_back_xyxy(*yoyo.getsize(), *c_image.shape[:2],
                                                                           *convertBack(predicts[idx][2]))]
                    ret_info['XMIN'] = a
                    ret_info['YMIN'] = b
                    ret_info['XMAX'] = c
                    ret_info['YMAX'] = d
                    ret_info['SCORE'] = predicts[idx][1]
                    ret_list.append(ret_info)
                # diff = (diff * 255).astype("uint8")
                # print("SSIM: {}".format(score))
    except:
        logger.error(traceback.format_exc())
        logger.error(exec)
        logger.debug(
            f'deal_detect: {frame.f_lineno}: cc is {predicts[idx]}, c_image.shape:{c_image.shape},predicts is {predicts}...')
        return None
    return ret_list


@app.route('/illpark', methods=['POST'])
def vehicle():
    device_info_list = request.json.get("DEVICE_INFO", [])
    if not len(device_info_list):
        return Response(
            json.dumps({"code": "000003", "msg": "input argument error", "ANALYSIS_RESULT": {}}, ensure_ascii=False),
            mimetype='application/json')
    ret_infos = []
    for dd in device_info_list:
        deviceSn = dd['DEVICE_SN']
        linea = Instances(**{
            "k": float(dd['K1'].strip()), "b": float(dd['B1'].strip())
        })
        lineb = Instances(**{
            "k": float(dd['K2'].strip()), "b": float(dd['B2'].strip())
        })

        img = base64toImageArray(dd['IMG_BASE64'])
        ts = dd['TIME_STAMP']
        ti = dd['TIME_INTERVAL']
        device_info = {}
        device_info['LINE_A'] = linea
        device_info['LINE_B'] = lineb
        dl = vehicle_device_infos.get(deviceSn)
        if dl:
            device_info['l_img'] = dl.get('c_img')
            device_info['l_ts'] = dl.get('c_ts')
            device_info['l_ti'] = dl.get('c_ti')
            device_info['l_objs'] = dl.get('c_objs')

        vehicle_device_infos[deviceSn] = device_info
        device_info['c_img'] = img
        device_info['c_ts'] = ts
        device_info['c_ti'] = ti
        if dl and (device_info['c_ts'] > device_info['l_ts'] + device_info['c_ti']) and len(device_info['l_objs']):
            xx = deal_detect(vehicle_device_infos[deviceSn])
            if xx: ret_infos.extend(xx)
        else:
            continue
    if len(ret_infos):
        ret = {
            "code": "000000",
            "msg": "succeed",
            "ANALYSIS_RESULT": ret_infos
        }
        return Response(json.dumps(ret, ensure_ascii=False), mimetype='application/json')
    else:
        return Response(json.dumps({"code": "000001", "msg": "succeed", "ANALYSIS_RESULT": {}}, ensure_ascii=False),
                        mimetype='application/json')
