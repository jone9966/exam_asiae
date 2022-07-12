import dlib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import glob


imgs = glob.glob('./image/faceon/raw/fake/faceon46.jpg')
for i, img in enumerate(imgs):
    detector = dlib.get_frontal_face_detector()
    img_cut = dlib.load_rgb_image(img)
    img_result = cv2.imread(img)
    try:
        face = detector(img_cut, 1)
        print(face)
    except:
        print(i)
        continue

    for idx, f in enumerate(face):
        (top, bottom) = f.top()-65, f.bottom()+10
        (left, rigth) = f.left()-30, f.right()+30
        face_in_img = img_result[top:bottom, left:rigth, :]
        try:
            cv2.imwrite('./image/faceon/raw/fake/cut/fake_{}.jpg'.format(idx+187), face_in_img)
        except:
            continue