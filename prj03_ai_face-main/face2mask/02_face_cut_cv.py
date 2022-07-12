import cv2
import cvlib as cv
import glob


imgs = glob.glob('./image/faceon/*.jpg')
for i, img in enumerate(imgs):
    img = cv2.imread(img)
    try:
        face, confidence = cv.detect_face(img, 0.8)
        if not face:
            continue
    except:
        print(i)
        continue

    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        face_in_img = img[startY:endY, startX:endX, :]
        cv2.imwrite('./image/faceon/face/mask_{}.jpg'.format(idx), face_in_img)