# coding: utf-8
import datetime
from math import ceil

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD

from keras_ssd512 import ssd_512
from keras_ssd_loss import SSDLoss
from ssd_batch_generator import BatchGenerator
from ssd_box_encode_decode_utils import SSDBoxEncoder


def train_hs512(lr=1e-4,
                freeze_bn=False,
                optim=None,
                batch_size=8,
                weights_path=None,
                save_weights_only=True,
                epochs=25):
    if weights_path is None:
        weights_path = 'VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'

    img_height = 512  # 1088 // 2  # Height of the input images
    img_width = 512  # 2048 // 2  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    # subtract_mean = [104, 117, 123] # The per-channel mean of the images in the dataset
    subtract_mean = [138, 138, 138]  # The per-channel mean of the images in the dataset
    swap_channels = False
    n_classes = 20  # The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
    scales = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]  # MS COCO scales
    # scales = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    # The space between two adjacent anchor box center points for each predictor layer.
    steps = [8, 16, 32, 64, 128, 256, 512]
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2, 0.2]
    coords = 'centroids'
    normalize_coords = True

    # 1: Build the Keras model
    K.clear_session()
    model, pred_sizes = ssd_512(image_size=(img_height, img_width, img_channels),
                                n_classes=n_classes,
                                l2_regularization=0.0005,
                                scales=scales,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                steps=steps,
                                offsets=offsets,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                coords=coords,
                                normalize_coords=normalize_coords,
                                subtract_mean=None,
                                divide_by_stddev=None,
                                swap_channels=swap_channels,
                                return_predictor_sizes=True)

    # 2: Load the trained VGG-16 weights into the model.
    model.load_weights(weights_path, by_name=True)

    # 3: Instantiate the optimizer and the SSD loss function and compile the model

    if optim is None:
        optim = SGD(lr=lr, momentum=0.9)
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    # 4: freeze the base model if needed
    if freeze_bn:
        for l in model.layers[:38]:
            l.trainable = False

    # 5: compile model
    model.compile(optimizer=optim, loss=ssd_loss.compute_loss)

    ## Prepare data generation

    # 1: Instantiate to `BatchGenerator` objects: One for training, one for validation.

    train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.

    images_path_root = './Datasets/'
    train_combined_labels = './Datasets/train_combined_ssd_512.txt'
    val_labels = './Datasets/val_ssd_512.txt'

    train_dataset.parse_csv(images_dir=images_path_root, labels_filename=train_combined_labels,
                            input_format=['image_name', 'class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    val_dataset.parse_csv(images_dir=images_path_root, labels_filename=val_labels,
                          input_format=['image_name', 'class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

    # 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=pred_sizes,
                                    min_scale=None,
                                    max_scale=None,
                                    scales=scales,
                                    aspect_ratios_global=None,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    limit_boxes=limit_boxes,
                                    variances=variances,
                                    pos_iou_threshold=0.5,
                                    neg_iou_threshold=0.2,
                                    coords=coords,
                                    normalize_coords=normalize_coords)

    # 4: Set the image processing / data augmentation options and create generator handles.

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             train=True,
                                             ssd_box_encoder=ssd_box_encoder,
                                             equalize=False,
                                             brightness=(0.5, 2, 0.5),
                                             flip=0.5,
                                             translate=False,
                                             scale=False,
                                             max_crop_and_resize=(img_height, img_width, 1, 3),
                                             # This one is important because the Pascal VOC images vary in size
                                             random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                             # This one is important because the Pascal VOC images vary in size
                                             random_crop=False,
                                             crop=False,
                                             resize=False,
                                             gray=True,
                                             limit_boxes=True,
                                             # While the anchor boxes are not being clipped, the ground truth boxes should be
                                             include_thresh=0.4)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3),
                                         # This one is important because the Pascal VOC images vary in size
                                         random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                         # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=True,
                                         limit_boxes=True,
                                         include_thresh=0.4)

    # Get the number of samples in the training and validations datasets to compute the epoch lengths below.
    n_train_samples = train_dataset.get_n_samples()
    n_val_samples = val_dataset.get_n_samples()

    # ## 4. Run the training

    fingerprint = 'ssd{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    tbCallBack = TensorBoard(log_dir='./Graph/{}'.format(fingerprint),
                             histogram_freq=0,
                             batch_size=batch_size,
                             write_graph=True)

    checkpointer = ModelCheckpoint('./saved/{{val_loss:.2f}}__{}_best_weights.h5'.format(fingerprint),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=save_weights_only,
                                   save_weights_only=True,
                                   mode='auto', period=1)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=5,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=1e-6)

    stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)

    epochs = 30

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=ceil(n_train_samples / batch_size),
                                  epochs=epochs,
                                  callbacks=[checkpointer,
                                             learning_rate_reduction,
                                             stopper,
                                             tbCallBack],
                                  validation_data=val_generator,
                                  validation_steps=ceil(n_val_samples / batch_size))

