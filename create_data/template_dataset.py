import cv2
import numpy as np
import os

def rotate_img(image, angle):
    # 画像の高さと幅
    h, w, c = image.shape[:3]
    h1, w1 = 84, 84

    center_x = (w - w1) // 2
    center_y = (h - h1) // 2
    cw = w // 2
    ch = h // 2
    center = (cw, ch)

    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # cos = np.abs(rot_matrix[0, 0])
    # sin = np.abs(rot_matrix[0, 1])

    # new_w = int((h*sin)+(w*cos))
    # new_h = int((h*cos)+(w*sin))

    # rot_matrix[0, 2] += (new_w/2) - center[0]
    # rot_matrix[1, 2] += (new_h/2) - center[1]

    # canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    # canvas[:, :] = (255, 255, 255)

    # rotated = cv2.warpAffine(image, rot_matrix, (new_w, new_h), borderValue=(255,255,255))
    rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
    rotated_trim = rotated[center_x:center_x + w1, center_y:center_y + h1]

    # background = cv2.warpAffine(image, rot_matrix, (new_w, new_h), borderValue=(0,0,0))
    # background = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # mask = np.all(rotated == [255,255,255], axis=-1)
    # mask = (rotated == 0)
    # rotated[mask] = background[mask]

    return rotated_trim

if __name__ == '__main__':
    path = '../new_dataset'
    os.makedirs(path, exist_ok=True)       
    for j in range(1,4):
        os.makedirs(path+f'/new_data{j}/', exist_ok=True)
        #image = cv2.imread(f'new_dataset/w_black/black_0.png', 1)
        #image = cv2.resize(image, (200,200), interpolation=cv2.INTER_LINEAR)
        image = cv2.imread(f'../image/new_data{j}.png', 1)
        
        number = 360
        for i in range(0,number):
            angle = i * 1.0
            rotate_image = rotate_img(image, angle)
            cv2.imwrite(f'../new_dataset/new_data{j}/new_data_'+str(i) +'.png', rotate_image)
