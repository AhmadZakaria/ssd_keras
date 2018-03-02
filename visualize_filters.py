from __future__ import print_function

import os
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from scipy.misc import imsave

from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from keras_ssd_loss import SSDLoss

# dimensions of the generated pictures for each filter.
img_width = 512
img_height = 512


# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
# layer_name = 'conv4_3'


# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def visualize_and_save_filters(model_path, name=None, layer_names=None):
    # build the VGG16 network with ImageNet weights
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    K.clear_session()  # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'compute_loss': ssd_loss.compute_loss})

    print('Model loaded.')

    model.summary()

    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    for layer_name in layer_names:
        directory = 'filters/{}/{}'.format(name, layer_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        kept_filters = []
        for filter_index in range(256):
            # we only scan through the first 200 filters,
            # but there are actually 512 of them
            print('Processing filter %d' % filter_index)
            start_time = time.time()

            # we build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            layer_output = layer_dict[layer_name].output
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some random noise
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, 3, img_width, img_height))
            else:
                input_img_data = np.random.random((1, img_width, img_height, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # we run gradient ascent for 20 steps
            for i in range(100):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    break

            # decode the resulting input image
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
            end_time = time.time()
            print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

        # we will stich the best 64 filters on a 8 x 8 grid.
        n = 8

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters

        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if idx >= len(kept_filters):
                    break

                # print(idx, len(kept_filters))
                img, loss = kept_filters[idx]
                imsave('{}/{}.png'.format(directory, str(idx)), img)
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

        # save the result to disk
        imsave('filters/filters_{}_{}.png'.format(name, layer_name), stitched_filters)


if __name__ == '__main__':
    layer_names = ['conv4_3', 'conv5_3', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv4_3_norm']
    visualize_and_save_filters('/home/iss/Dokumente/new_ssd_keras/original_model.h5', name='VOC0712plus',
                               layer_names=layer_names)

    visualize_and_save_filters('/home/iss/Dokumente/new_ssd_keras/best_model.h5', name='adam_final',
                               layer_names=layer_names)
