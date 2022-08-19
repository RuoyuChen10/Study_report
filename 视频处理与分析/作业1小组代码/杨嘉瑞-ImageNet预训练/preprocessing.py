import numpy as np
import os
import json
import cv2


# 混乱：0； 空洞：1； 分裂：2； 受伤：3； 流动：4； 趋中：5； 整合：6； 能量：7
def parse_json(json_file):
    data = json.load(open(json_file, 'r', encoding='utf-8'))['themes']
    label = np.zeros(8)
    label[[data[j]['type'] for j in range(len(data))]] = 1
    return label

def cal_mean_std(root_path):
    image_list = os.listdir(root_path)
    R_mean, G_mean, B_mean = [], [], []
    for image in image_list:
        image_data = cv2.imread(os.path.join(root_path, image, 'BireView.png'))
        R_mean.append(np.mean(image_data[:, :, 2]))
        G_mean.append(np.mean(image_data[:, :, 1]))
        B_mean.append(np.mean(image_data[:, :, 0]))
    aver_R = np.mean(R_mean) / 255.0
    aver_G = np.mean(G_mean) / 255.0
    aver_B = np.mean(B_mean) / 255.0
    print('the average of R-G-B of dataset is : {0:.3f} - {1:.3f} - {2:.3f}'.format(aver_R, aver_G, aver_B))

    std_R, std_G, std_B = 0.0, 0.0, 0.0
    for image in image_list:
        image_data = cv2.imread(os.path.join(root_path, image, 'BireView.png'))
        h, w, _ = image_data.shape
        image_R = image_data[:, :, 2] / 255.0
        image_G = image_data[:, :, 1] / 255.0
        image_B = image_data[:, :, 0] / 255.0
        std_R += np.sum(np.power(image_R - aver_R, 2)) / (w * h)
        std_G += np.sum(np.power(image_G - aver_G, 2)) / (w * h)
        std_B += np.sum(np.power(image_B - aver_B, 2)) / (w * h)
    std_R = np.sqrt(std_R / len(image_list))
    std_G = np.sqrt(std_G / len(image_list))
    std_B = np.sqrt(std_B / len(image_list))
    print('the standard deviation of R-G-B of dataset is : {0:.3f} - {1:.3f} - {2:.3f}'.format(std_R, std_G, std_B))


if __name__ == '__main__':
    root_path = '作业一数据/课程实验数据'
    cal_mean_std(root_path)
    

