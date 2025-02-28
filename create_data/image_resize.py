import cv2

for i in range(1, 5):
    image = cv2.imread(f'../image/new_data{i}.png')
    image_resize = cv2.resize(image, (84, 84))
    cv2.imwrite(f'../image/new_resize{i}.png', image_resize)