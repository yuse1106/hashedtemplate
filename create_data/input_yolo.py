import cv2
import numpy as np
import glob
import os
import random

def check(new_x, new_y, object_width, object_height, placed_objects):
    # for(x, y, width, height, label) in placed_objects:
    #     if(new_x < x + width and new_x + object_x > x and
    #        new_y < y + height and new_y + object_y > y):
    #         return True
    #     return False
    for label, x, y, width, height in placed_objects:
        if(new_x < x + width and new_x + object_width > x and
           new_y < y + height and new_y + object_height > y):
            return False
    return True


if __name__ == '__main__':
    image_path = '../../yolov8/datasets/images/'
    label_path = '../../yolov8/datasets/labels/'
    os.makedirs(image_path+'test/', exist_ok=True)
    os.makedirs(label_path+'test/', exist_ok=True)
    location = []
    for i in range(20):
        # 背景画像のサイズを設定
        height, width = 416, 416
        # 背景画像を真っ白に，＋することにより画像の色を設定
        back_img = np.zeros((height, width, 3), dtype=np.uint8)
        back_img += 255

        path = '../tra1'
        path_list = glob.glob(path+'/*')
        # 配置済みの位置
        placed_objects = []
        # YOLOのラベルファイル
        yolo_label = []
        # ラベルごとに配置する物体の数
        object_num = 3

        for label, pic_path in enumerate(path_list):
            list = glob.glob(pic_path+'/*')
            for j in range(object_num):
                data = random.choice(list)
                select_img = cv2.imread(data)
                # 物体画像のサイズ
                object_height, object_width, c = select_img.shape
                # print(object_height, object_width, c)

                # 物体が重ならないように配置
                placed = False
                while not placed:
                    x_position = random.randint(0, width - object_width) 
                    y_position = random.randint(0, height - object_height) 

                    x_position1 = x_position / width
                    y_position1 = y_position / height

                    object_width1 = object_width / width
                    object_height1 = object_height / height

                    if check(x_position, y_position, object_width, object_height, placed_objects):
                        break
                placed_objects.append((label, x_position, y_position, object_width, object_height))

                x_center = (x_position+x_position+object_height) / 2 / width
                y_center = (y_position+y_position+object_width) / 2 / height
                yolo_width = object_width / width
                yolo_height = object_height / height
                yolo_label.append((int(label), x_center, y_center, yolo_width, yolo_height))

                back_img[y_position:y_position+object_height, x_position:x_position+object_width] = select_img

        placed_objects = np.array(placed_objects)   
        sort_index = np.lexsort((placed_objects[:,1], placed_objects[:,2]))
        placed_objects_sort = placed_objects[sort_index]
        location.append(placed_objects_sort) 
        yolo_label = np.array(yolo_label)
        y_sort = np.lexsort((yolo_label[:,1], yolo_label[:,2]))  
        yolo_sort = yolo_label[y_sort]
        cv2.imwrite(f'../test/input2/input_{i}.jpg', back_img)
        np.savetxt(f'../../yolov8/location_test/location{i}.txt', yolo_sort, fmt='%d %.6f %.6f %.6f %.6f')  
        np.savetxt(label_path+f'test/image{i}.txt', yolo_sort, fmt='%d %.6f %.6f %.6f %.6f')
        cv2.imwrite(image_path+f'test/image{i}.jpg', back_img)
    np.save('../../yolov8/npy/yolo_location.npy', yolo_sort)
    np.save('../simple/npy/yolo_location.npy', location)
