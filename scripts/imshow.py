import cv2 as cv
import os
import torch
import numpy as np


def imshow(train_or_valid):
    global files, path, label_path
    if train_or_valid == "t":
        files = train_files
        image_path = train_image_path
        label_path = train_label_path
    elif train_or_valid == "v":
        files = valid_files
        image_path = valid_image_path
        label_path = valid_label_path

    for i, file in enumerate(files):
        # 获得图片对应的label文件名
        label = file.split(".jpg")[0]
        label = np.loadtxt(label_path + "\\" + label + ".txt")
        label = torch.from_numpy(label)
        label_sliced = label[-label.shape[0] + 1:]  # xywh
        if label.shape[0] != 0:
            label_sliced_ = torch.zeros(4)
            label_sliced_[0] = label_sliced[0] - label_sliced[2] / 2  # x1
            label_sliced_[2] = label_sliced[0] + label_sliced[2] / 2  # y1
            label_sliced_[1] = label_sliced[1] - label_sliced[3] / 2  # y2
            label_sliced_[3] = label_sliced[1] + label_sliced[3] / 2  # y2
        else:
            label_sliced_ = torch.zeros(4)

        # 读取训练集图片
        image = cv.imread(image_path + "\\" + file)
        color_image = image
        print(color_image.shape)
        image_norm = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        image_t = torch.from_numpy(np.array(image_norm, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        pred = model(image_t.to('cuda')).squeeze().to('cpu').float()
        print(pred)
        x1 = int(pred[0] * 640)
        y1 = int(pred[1] * 480)
        x2 = int(pred[2] * 640)
        y2 = int(pred[3] * 480)

        image = cv.resize(image, (640, 480))
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.putText(image, "prediction", (x1, y1 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        x1_g = int(label_sliced_[0] * 640)
        y1_g = int(label_sliced_[1] * 480)
        x2_g = int(label_sliced_[2] * 640)
        y2_g = int(label_sliced_[3] * 480)

        cv.rectangle(image, (x1_g, y1_g), (x2_g, y2_g), (255, 0, 255), 2)
        cv.putText(image, "label", (x1_g, y1_g + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # 显示图像
        cv.imshow('Image with anchor box', image)
        cv.waitKey(1000)


if __name__ == "__main__":
    version = "1"
    # 读入图像路径
    train_image_path = r"..\robot_simulation-" + version + r"\train\images"
    train_label_path = r"..\robot_simulation-" + version + r"\train\labels"
    valid_image_path = r"..\robot_simulation-" + version + r"\valid\images"
    valid_label_path = r"..\robot_simulation-" + version + r"\valid\labels"

    train_files = os.listdir(train_image_path)
    train_size = len(train_files)
    valid_files = os.listdir(valid_image_path)
    valid_size = len(valid_files)

    model = torch.load("model_csp3_1000.pth")
    imshow("v")
