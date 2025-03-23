import random

import numpy as np
import os
from PIL import Image

np.random.seed(42)


class ORLDataset:
    def __init__(self, image_path):
        self.image_path = image_path
        self.total_images_list = []

    def load_data(self):
        """
        :return: return List[], include 40 people's 10 images with total mean and var
        """
        for folder_name in sorted(os.listdir(self.image_path)):
            folder_path = os.path.join(self.image_path, folder_name)

            if os.path.isdir(folder_path):
                image_list = []

                for file_name in sorted(os.listdir(folder_path)):
                    if file_name.endswith('.pgm'):
                        file_path = os.path.join(folder_path, file_name)
                        image = Image.open(file_path)
                        image_array = np.array(image, dtype=np.float32)
                        image_list.append(image_array)
                self.total_images_list.append(image_list)

        m_v_count = np.vstack(self.total_images_list)
        mean = np.mean(m_v_count)
        var = np.var(m_v_count)

        self.preprocess(mean, var)

        return self.total_images_list

    def preprocess(self, mean, var):
        for person_images in self.total_images_list:
            for i, image in enumerate(person_images):
                # Histogram + Normalized
                hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized * 255).reshape(image.shape)
                person_images[i] = (image_equalized.astype(np.float32) - mean) / np.sqrt(var)
                person_images[i] = (2 * (person_images[i] - np.min(person_images[i])) /
                                    (np.max(person_images[i]) - np.min(person_images[i])) - 1)
                # TODO: use [-1,1] or [0,1] ?
                # person_images[i] = ((person_images[i] - np.min(person_images[i])) /
                #                     (np.max(person_images[i]) - np.min(person_images[i])))

    def split_personal_image(self, train_image_number):
        train_set = []
        train_label = []
        test_set = []
        test_label = []

        for label, person_images in enumerate(self.total_images_list):
            random.shuffle(person_images)

            train_set.append(person_images[:train_image_number])
            train_label.append([label] * train_image_number)
            test_set.append(person_images[train_image_number:])
            test_label.append([label] * (10 - train_image_number))

        return (np.array(train_set), np.array(train_label),
                np.array(test_set), np.array(test_label))


if __name__ == '__main__':
    pass
