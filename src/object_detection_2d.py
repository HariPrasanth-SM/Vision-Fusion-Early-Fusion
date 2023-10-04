from yolov4.tf import YOLOv4
import tensorflow as tf
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

yolo = YOLOv4(tiny=False)
yolo.classes = "../data/Yolov4/coco.names"
yolo.make_model()
yolo.load_weights("../data/Yolov4/yolov4.weights", weights_type="yolo")

def run_obstacle_detection(img):
    start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
    resized_image = resized_image / 255.
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)

    _candidates = []
    result = img.copy()
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
        candidates = np.concatenate(_candidates, axis=1)
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] #https://stackoverflow.com/questions/35673095/python-how-to-eliminate-all-the-zero-rows-from-a-matrix-in-numpy?lq=1
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        exec_time = time.time() - start_time
        result = yolo.draw_bboxes(img, pred_bboxes)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, pred_bboxes

if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("../data/img/000031.png"), cv2.COLOR_BGR2RGB)
    result, pred_bboxes = run_obstacle_detection(image)

    fig_camera = plt.figure(figsize=(14, 7))
    ax_lidar = fig_camera.subplots()
    ax_lidar.imshow(result)
    plt.savefig("../output/2d_object_detection.png")
    plt.show()