import cv2
import os
import glob
import random 
import numpy as np

def check(new_x, new_y, object_x, object_y, placed_objects):
    for label, x, y, width, height in placed_objects:
        if(new_x < x + width and new_x + object_x > x and
           new_y < y + height and new_y + object_y > y):
            return False
    return True
    
if __name__ == '__main__':
    location = []
    for i in range(20):
        # 背景画像のサイズ
        back_height, back_width = 416, 416
        # 背景画像を真っ白に
        back_img = np.zeros((back_height, back_width, 3), dtype=np.uint8)
        back_img += 255
        #array = np.array(back_img)
        #print(array)
        cv2.imwrite('../image/input_back.png', back_img)

        path1 = '../tra1'
        path_list = glob.glob(path1+'/*')
        # 配置済みの位置
        placed_object = []
        # placed_objects = {}
        yolo_label = []
    
        # pic_path = ('tra/orange')
        for label, pic_path in enumerate(path_list):
            image_list = glob.glob(pic_path+'/*')
            for j in range(3):
                data = random.choice(image_list)
                select_img = cv2.imread(data)
                # 物体画像のサイズ
                object_height, object_width, c = select_img.shape

                # 物体が重ならないように配置
                placed = False
                while not placed:
                    x_position = random.randint(0, back_width - object_width)
                    y_position = random.randint(0, back_height - object_height)

                    # x_position1 = x_position / back_width
                    # y_position1 = y_position / back_height

                    # object_width1 = object_width / back_width
                    # object_height1 = object_height / back_height

                    # 重なり
                    if check(x_position, y_position, object_width, object_height, placed_object):
                        break
                    # if check(x_position1, y_position1, object_width1, object_height1, placed_object):
                        # break
                # 物体の位置追加
                placed_object.append((label, x_position, y_position, object_width, object_height))
                # placed_object.append((label, x_position1, y_position1, object_width1, object_height1))
                # placed_object.append((label, x_position, y_position))

                x_center = (x_position+x_position+object_width) /2 / back_width
                y_center = (y_position+y_position+object_height) / 2 / back_height
                yolo_width = object_width / back_width
                yolo_height = object_height / back_height
                yolo_label.append((label, x_center, y_center, yolo_width, yolo_height))
                # 背景画像に物体を配置する
                back_img[y_position:y_position + object_height, x_position:x_position + object_width] = select_img

        placed_objects = np.array(placed_object)
        sort_index = np.lexsort((placed_objects[:,1], placed_objects[:,2]))
        placed_objects_sort = placed_objects[sort_index]
        location.append(placed_objects_sort)
        yolo_label = np.array(yolo_label)
        y_sort = np.lexsort((yolo_label[:,1], yolo_label[:,2]))
        yolo_sort = yolo_label[y_sort]
        np.savetxt(f'../../yolov8/location_jpg/location{i}.txt', yolo_sort, fmt='%6f')
        np.savetxt(f'../../yolov8/datasets/labels/test/input_{i}.txt', yolo_sort, fmt='%6f')
        # print(location)
        # np.save('newdata/newdata_npy/placed_objects.npy', placed_objects_sort)
        # cv2.imshow("out", back_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f'../test/simple/color_jpg/input_{i}.jpg', back_img)
        cv2.imwrite(f'../../yolov8/datasets/images/test/input_{i}.jpg', back_img)
        #     # ディレクトリ内の物体画像のファイルリストを取得
        # object_images_list = [f for f in os.listdir(objects_directory) if os.path.isfile(os.path.join(objects_directory, f))]
    np.save('../simple/npy/location_jpg.npy', location)
    print(location)
    # print('a',location[0])