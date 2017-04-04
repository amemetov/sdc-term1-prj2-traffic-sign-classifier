import numpy as np
import skimage
import skimage.transform as transform
from sklearn.utils import shuffle

class AugmentDataGenerator(object):
    def __init__(self, X_origin, y_origin):
        self.X_origin = X_origin
        self.y_origin = y_origin

    def next_batch(self, batch_size):
        choice_ind = np.random.choice(self.X_origin.shape[0], batch_size)
        X = self.X_origin[choice_ind]
        y = self.y_origin[choice_ind]

        X_batch = np.ndarray(X.shape)

        for i in range(X.shape[0]):
            if np.random.random() < 1:#0.5:
                X_batch[i] = augment_image(X[i])
            else:
                X_batch[i] = X[i]

        return X_batch, y


def make_noisy_image(image):
    #noisy = image + 10 * np.random.random(image.shape)
    #return noisy.astype(np.uint8)
    #modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    modes = ['poisson', 'speckle']
    mode = modes[np.random.randint(0, len(modes))]
    return skimage.util.random_noise(image, mode=mode)

def zoom_out_image(image):
    scale = np.random.uniform(low=0.5, high=0.9)
    scaled = transform.rescale(image, scale)
    dw = int((image.shape[0] - scaled.shape[0]) / 2)
    dh = int((image.shape[1] - scaled.shape[1]) / 2)
    padded = np.pad(scaled, ((dh,dh), (dw,dw), (0,0)), 'constant', constant_values=0)
    return transform.resize(padded, image.shape)

def zoom_in_image(image):
    scale = np.random.uniform(low=1.1, high=2.0)
    scaled = transform.rescale(image, scale)
    #crop center of the scaled image
    dx0 = int((scaled.shape[0] - image.shape[0]) / 2)
    dx1 = scaled.shape[0] - dx0
    dy0 = int((scaled.shape[1] - image.shape[1]) / 2)
    dy1 = scaled.shape[1] - dy0
    cropped = scaled[dx0:dx1, dy0:dy1, :]
    return transform.resize(cropped, image.shape)

def rotate_image(image):
    angle = np.random.randint(-25, 25)
    return transform.rotate(image, angle)

def shift_image(image):
    shift_val = np.random.randint(-5, 5)
    return np.roll(image, [shift_val, shift_val, shift_val])


def augment_image(image, combine_prob=0.5):
    # methods = [make_noisy_image, zoom_out_image, zoom_in_image, rotate_image, shift_image]
    methods = [zoom_out_image, zoom_in_image, rotate_image, shift_image]
    method = methods[np.random.randint(0, len(methods))]
    image = method(image)
    # combine augment methods
    if np.random.random() <= combine_prob:
        # decrease combine_prob
        return augment_image(image, combine_prob=combine_prob-0.25)
    else:
        return image


def augment_data(X, y, n_classes, target_total_samples_count):
    # Balance data by adding augment samples

    target_items_per_class = int(target_total_samples_count / n_classes) + 1
    overflow_items_num = 0

    # Group by class
    X_by_class = np.array([None] * n_classes)
    for c in range(n_classes):
        X_by_class[c] = X[y == c]
        if X_by_class[c].shape[0] > target_items_per_class:
            overflow_items_num += X_by_class[c].shape[0] - target_items_per_class

    target_total_samples_count += overflow_items_num

    X_result = np.zeros((target_total_samples_count, X.shape[1], X.shape[2], X.shape[3]))
    X_result[:X.shape[0]] = X
    y_result = np.zeros(target_total_samples_count)
    y_result[:y.shape[0]] = y

    result_idx = X.shape[0]

    for c in range(n_classes):
        print("============ Processing class {} ============".format(c))
        items_per_class = len(X_by_class[c])
        if items_per_class < target_items_per_class:
            num_augment_items = target_items_per_class - items_per_class
            print("--- Need to add {0} augment samples ---".format(num_augment_items))
            images = X_by_class[0][np.random.choice(X_by_class[0].shape[0], num_augment_items)]
            for image in images:
                if result_idx >= target_total_samples_count:
                    break
                augment_sample = augment_image(image)
                X_result[result_idx] = augment_sample
                y_result[result_idx] = c
                result_idx += 1

    print("Result idx: {}, Total Samples Count {}".format(result_idx, target_total_samples_count))
    X_result, y_result = shuffle(X_result, y_result)
    return (X_result, y_result)
