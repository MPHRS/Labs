import os
import skimage
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2

def second(image):
    # Преобразование изображения в массив numpy
    arr = io.imread(image)

    # Генерация 1000 случайных координат и цветов
    for _ in range(1000):
        color_t = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        x_t = np.random.randint(0, arr.shape[0])
        y_t = np.random.randint(0, arr.shape[1])
        arr[x_t][y_t] = color_t
    io.imsave(os.path.join(os.getcwd(),"files_for_lab", "lab3", "output", "task_2.jpg"), arr)
    # Отображение изображения с добавленными точками
    plt.imshow(arr)
    plt.show()



# io.imshow(img)
# io.show()

def third(img):
    def threshold_image(image, threshold_value):
        # Преобразование изображения в оттенки серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применение порогового значения
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        return thresholded_image

    def on_trackbar(val):
        global threshold_value
        threshold_value = val
        thresholded_image = threshold_image(image, threshold_value)
        cv2.imshow('Thresholded Image', thresholded_image)

    # Загрузка изображения
    image = cv2.imread(img)


    # Создание окна для отображения изображения
    cv2.namedWindow('Original Image')
    cv2.imshow('Original Image', image)

    # Создание окна для отображения черно-белой версии изображения
    cv2.namedWindow('Thresholded Image')

    # Создание Trackbar'а
    threshold_value = 128
    cv2.createTrackbar('Threshold', 'Thresholded Image', threshold_value, 255, on_trackbar)

    # Отображение изображений и ожидание нажатия клавиши
    cv2.waitKey(0)

    # Уничтожение окон
    cv2.destroyAllWindows()
    io.imsave(os.path.join(os.getcwd(),"files_for_lab", "lab3", "output", "task_3.jpg"), threshold_image(image, threshold_value))
    # Отображение изображения с добавленными точками
    plt.imshow(threshold_image(image, cv2.waitKey(0)))
    plt.show()



def fourth(img, channel_idx):
    image = cv2.imread(img)
    # разбиваем изображение на отдельные каналы

    b, g, r = cv2.split(image)

    # создаем пустое изображение той же формы, что и исходное изображение
    channel_image = np.zeros_like(image)

    # выбираем нужный канал и оставляем его на изображении
    if channel_idx == 0:
        channel_image[:, :, 0] = r
    elif channel_idx == 1:
        channel_image[:, :, 1] = g
    elif channel_idx == 2:
        channel_image[:, :, 2] = b

    io.imsave(os.path.join(os.getcwd(),"files_for_lab", "lab3", "output", "task_4.jpg"), channel_image)
    # Отображение изображения с добавленными точками
    plt.imshow(channel_image)
    plt.show()

def twentieh(img):
    image = cv2.imread(img)

    # Разбиваем изображение на четыре части
    height, width, _ = image.shape
    half_h, half_w = height // 2, width // 2
    parts = [
        image[:half_h, :half_w],
        image[half_h:, :half_w],
        image[:half_h, half_w:],
        image[half_h:, half_w:]
    ]

    # Оставляем только один канал в каждой части
    result = np.zeros_like(image)
    for i, part in enumerate(parts):
        if i < 3:
            b, g, r = cv2.split(part)
            if i == 0:
                result[:half_h, :half_w, 0] = b
            elif i == 1:
                result[half_h:, :half_w, 1] = g
            elif i == 2:
                result[:half_h, half_w:, 2] = r
        else:
            gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
            result[half_h:, half_w:, :] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    io.imsave(os.path.join(os.getcwd(),"files_for_lab", "lab3", "output", "task_20.jpg"), result)
    # Отображение изображения с добавленными точками
    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    img = os.path.join(os.getcwd(),"files_for_lab", "lab3", "input", "img.jpg")
    second(img)
    third(img)
    fourth(img, 1)
    twentieh(img)