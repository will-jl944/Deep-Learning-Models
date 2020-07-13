import tensorflow as tf
import numpy as np
from tqdm import tqdm
from config import config
from utils.losses import dice_coe, mse_loss
import matplotlib.pyplot as plt
import skimage.io
import cv2


class Train_manager():
    def __init__(self, Regressor_class, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.validation_dataset_it = iter(test_dataset)
        self.model_replica_list = []
        # create one replica of model on each tower
        for gpu_id in range(config.num_gpu):
            with tf.device('/device:GPU:{}'.format(gpu_id)):
                with tf.name_scope('GPU_{}'.format(gpu_id)):
                    self.model_replica_list.append(Regressor_class())

        with tf.name_scope('Adam_Optimizer'):
            self.optimizer = tf.optimizers.Adam(config.lr)

        self.train_writer = tf.summary.create_file_writer('{}/train'.format(config.log_dir))
        self.validation_writer = tf.summary.create_file_writer('{}/validation'.format(config.log_dir))

        self.graph_exported = False

    def fit(self):
        for epoch in range(config.total_epoch):
            for step, element in tqdm(enumerate(self.train_dataset)):
                self.train_step(element)

                if step % 50 == 0:
                    validation_element = next(self.validation_dataset_it)
                    img, coord, validation_pred_coord, validation_loss = self.validation_step(validation_element)

                    image_with_truth_coord = self.draw_points(tf.image.grayscale_to_rgb(img).numpy(), coord, is_truth=True)

                    image_with_pred_coord = self.draw_points(tf.image.grayscale_to_rgb(img).numpy(), validation_pred_coord, is_truth=False)

                    # skimage.io.imsave('./validation/{}.tif'.format(step), validation_pred_pec)
                    with self.validation_writer.as_default():
                        tf.summary.image('Validation_Sample', image_with_pred_coord, max_outputs=1,
                                         step=self.optimizer.iterations)
                        tf.summary.image('Truth', image_with_truth_coord, max_outputs=1,
                                         step=self.optimizer.iterations)

                if step % 100 == 0:
                    self.model_replica_list[0].save_weights(
                        '{}/epoch_{}_iter_{}/model_{}'.format(config.model_dir, epoch, step, step),
                        overwrite=False)
            self.model_replica_list[0].save_weights(
                '{}/epoch_{}_iter_{}/model_{}'.format(config.model_dir, epoch, step, step),
                overwrite=False)

    def draw_points(self, image_batch, coord_batch, is_truth=True):
        if is_truth:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]
        batch_size, img_h, img_w, img_c = image_batch.shape
        _, coord_vector_len = coord_batch.shape

        output_image_batch = np.zeros(image_batch.shape)

        for batch_num in range(batch_size):
            image = image_batch[batch_num, ...]
            coord = coord_batch[batch_num, ...]
            x = coord[0]
            y = coord[1]
            image = cv2.circle(image, (x, y), radius=5, color=color, thickness=-1)
            output_image_batch[batch_num, ...] = image

        return output_image_batch

    @tf.function
    def validation_step(self, element):
        with tf.name_scope('Split_Data'):
            img = element[0]
            coord = element[2]

            # repeat channel to use pre-trained model
            # image = tf.tile(image, multiples=[1, 1, 1, 3])

            img_split = tf.split(img, config.num_gpu, axis=0)
            coord_split = tf.split(coord, config.num_gpu, axis=0)

        tower_coords = []
        tower_loss = []
        for gpu_id in range(config.num_gpu):
            with tf.device('/device:GPU:{}'.format(gpu_id)), tf.name_scope('GPU_{}'.format(gpu_id)):
                curr_tower_model = self.model_replica_list[gpu_id]
                pred_coord = curr_tower_model(img_split[gpu_id])
                loss = mse_loss(coord_split[gpu_id], pred_coord)
                # loss = dice_coe(pred_pec, pec_split[gpu_id])

                tower_coords.append(pred_coord);
                tower_loss.append(loss)

        with tf.device("/device:{}:0".format(config.controller)), tf.name_scope('Controller_Ops'):
            with tf.name_scope('Output'):
                pred_coord = tf.concat(tower_coords, axis=0)
                loss = tf.reduce_mean(tower_loss)

            with tf.name_scope('Summaries'):
                with self.validation_writer.as_default():
                    tf.summary.scalar("Total Loss", loss, step=self.optimizer.iterations)

        return img, coord, pred_coord, loss

    @tf.function
    def train_step(self, element):
        with tf.name_scope('Split_Data'):
            img = element[0]
            coord = element[2]
            img_split = tf.split(img, config.num_gpu, axis=0)
            coord_split = tf.split(coord, config.num_gpu, axis=0)

        tower_coords = []
        tower_loss = []
        tower_grads = []
        for gpu_id in range(config.num_gpu):
            with tf.device('/device:GPU:{}'.format(gpu_id)), tf.name_scope('GPU_{}'.format(gpu_id)):
                with tf.GradientTape() as tape:
                    curr_tower_model = self.model_replica_list[gpu_id]

                    pred_coord = curr_tower_model(img_split[gpu_id])
                    loss = mse_loss(coord_split[gpu_id], pred_coord)
                    # loss = dice_coe(pred_pec, pec_split[gpu_id])

                    tower_coords.append(pred_coord);
                    tower_loss.append(loss)

                    with tf.name_scope('Compute_Gradients'):
                        tower_grads.append(tape.gradient(tower_loss[-1], curr_tower_model.trainable_variables))

        with tf.device("/device:{}:0".format(config.controller)), tf.name_scope('Controller_Ops'):
            with tf.name_scope('Output'):
                pred_coord = tf.concat(tower_coords, axis=0)
                loss = tf.reduce_mean(tower_loss)

            with tf.name_scope('Summaries'):
                with self.train_writer.as_default():
                    tf.summary.scalar("Total Loss", loss, step=self.optimizer.iterations)

            # average gradients on controller device
            self.avg_grads = self.average_gradients(tower_grads)

        # apply averaged gradients to each tower weights
        with tf.name_scope('Update_Ops'):
            for gpu_id in range(config.num_gpu):
                with tf.device('/device:GPU:{}'.format(gpu_id)), tf.name_scope('GPU_{}'.format(gpu_id)):
                    curr_tower_model = self.model_replica_list[gpu_id]

                    # apply gradient on controller device
                    self.optimizer.apply_gradients(zip(self.avg_grads, curr_tower_model.trainable_variables))

        if not self.graph_exported:
            tf.compat.v1.summary.FileWriter("{}/model_graph".format(config.log_dir),
                                            graph=tf.compat.v1.get_default_graph())
            self.graph_exported = True
            print('Exporting Graph')

    def average_gradients(self, tower_grads):
        with tf.name_scope('Average_Gradients'):
            average_grads = []
            for tower_grad in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = [g for g in tower_grad]
                grad = tf.reduce_mean(grads, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.

                average_grads.append(grad)
            return average_grads
