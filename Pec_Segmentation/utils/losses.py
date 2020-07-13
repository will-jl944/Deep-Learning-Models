import tensorflow as tf
from config import config


def yolo_loss(grid, grid_label):
    '''

    :grid: (?, 2*GRID_N, 2*GRID_N, BOX_VECTOR_LEN) ---> BOX_VECTOR_LEN: [P, x, y, w, h, isMass, isCals]
    :return:
    '''
    with tf.name_scope('Loss'):
        # print('Loss_Grid:', grid)
        # print('Loss_Grid_Label:', grid_label)

        # print('grid_label:', grid_label)
        with tf.name_scope('Has_Obj_Logits'):
            has_obj_slice_logits = grid[:, :, :, 0]

        with tf.name_scope('Has_Obj_Label'):
            has_obj_slice_label = grid_label[:, :, :, 0]
        # iou = IoU(grid, grid_label)


        # obj loss - punish false negative
        with tf.name_scope('Obj_Loss'):
            obj_loss = (
            tf.reduce_sum(tf.multiply(tf.square(has_obj_slice_logits - has_obj_slice_label), has_obj_slice_label)))
            # print('obj_loss:',obj_loss)

        # no obj loss - punish false positive
        with tf.name_scope('No_Obj_Loss'):
            no_obj_loss = (tf.reduce_sum(tf.multiply(tf.square(has_obj_slice_logits), 1 - has_obj_slice_label)))
            # print('no_obj_loss:', no_obj_loss)

        # x, y, w, h regression loss
        with tf.name_scope('Box_Regression_Loss'):
            x_slice = grid[:, :, :, 1]
            x_slice_label = grid_label[:, :, :, 1]
            y_slice = grid[:, :, :, 2]
            y_slice_label = grid_label[:, :, :, 2]

            x_delta = tf.multiply(tf.square(x_slice - x_slice_label), has_obj_slice_label)
            y_delta = tf.multiply(tf.square(y_slice - y_slice_label), has_obj_slice_label)
            xy_coordinate_loss = (tf.reduce_sum(x_delta + y_delta))
            # print('xy_coordinate_loss', xy_coordinate_loss)

            w_slice = grid[:, :, :, 3]
            w_slice_label = grid_label[:, :, :, 3]

            h_slice = grid[:, :, :, 4]
            h_slice_label = grid_label[:, :, :, 4]
            # print(w_slice)
            w_delta = tf.multiply(tf.square(tf.sqrt(w_slice + config.EPS) - tf.sqrt(w_slice_label + config.EPS)), has_obj_slice_label)
            h_delta = tf.multiply(tf.square(tf.sqrt(h_slice + config.EPS) - tf.sqrt(h_slice_label + config.EPS)), has_obj_slice_label)
            wh_coordinate_loss = (tf.reduce_sum(w_delta + h_delta))
            # print('xy_coordinate_loss', wh_coordinate_loss)

        with tf.name_scope('Sum_Yolo_Loss_Components'):
            loss_component_dict = {'obj_loss': obj_loss, 'no_obj_loss': config.no_obj_coff*no_obj_loss,
                                   'xy_loss': config.coord_coff * xy_coordinate_loss,
                                   'wh_loss': config.coord_coff * wh_coordinate_loss}

            loss = 0
            for loss_component_name in loss_component_dict:
                loss += loss_component_dict[loss_component_name]

        return loss, loss_component_dict

def yolo_loss_wrapper(y_true, y_pred):
    loss, _ = yolo_loss(y_pred, y_true)
    return loss

def mse_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(y_true - y_pred), axis=[1,2,3]))

    return loss


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

