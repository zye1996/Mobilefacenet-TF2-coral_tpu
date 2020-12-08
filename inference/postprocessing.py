import cv2
import numpy as np
import scipy.special

from anchors import ANCHOR

OBJ_THRES = 0.7
NMS_THRES = 0.4
VARIANCE = [0.1, 0.2]
FACE_DIMENSION = [96, 112]

TEMPLATE = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393],
                       [0.500225, 0.64050536], [0.37097589, 0.82469196],
                       [0.631517, 0.82325089]])


def box_iou(box_1, box_2):

    # calculate area
    box_1_area = box_1[..., 2] * box_1[..., 3]
    box_2_area = box_2[..., 2] * box_2[..., 3]

    # calculate intersection coordinate
    l1 = box_1[..., 0] - box_1[..., 2] * 0.5
    l2 = box_2[..., 0] - box_2[..., 2] * 0.5
    left = np.maximum(l1, l2)
    r1 = box_1[..., 0] + box_1[..., 2] * 0.5
    r2 = box_2[..., 0] + box_2[..., 2] * 0.5
    right = np.minimum(r1, r2)
    bottom1 = box_1[..., 1] - box_1[..., 3] * 0.5
    bottom2 = box_2[..., 1] - box_2[..., 3] * 0.5
    bottom = np.minimum(bottom1, bottom2)
    t1 = box_1[..., 1] + box_1[..., 3] * 0.5
    t2 = box_2[..., 1] + box_2[..., 3] * 0.5
    top = np.maximum(t1, t2)

    w = right - left; h = top - bottom
    print(w, h)
    if w.all() > 0 and h.all() > 0:
        return (w * h) / (box_1_area + box_2_area - w * h)


def pred_boxes(box_output, score_output, ldmk_output):
    '''
    generate box information from output
    :param box_output: 3160*4
    :param score_output: 3160*2
    :param ldmk_output: 3160*10
    :return:
    '''

    # select boxes greater than threshold probability
    prob = scipy.special.softmax(score_output, axis=-1)
    pre_select_boxes_mask = prob[:, 1] > OBJ_THRES
    pre_select_boxes_index = np.where(pre_select_boxes_mask)[0]
    pre_select_anchor = ANCHOR[pre_select_boxes_index, :]

    # calculate coordinate
    box_cord = box_output[pre_select_boxes_index, :]
    box_cord = np.concatenate((
        pre_select_anchor[:, :2] + box_cord[:, :2] * VARIANCE[0] * pre_select_anchor[:, 2:],
        pre_select_anchor[:, 2:] * np.exp(box_cord[:, 2:] * VARIANCE[1])), 1)
    box_cord[:, :2] -= box_cord[:, 2:] / 2
    box_cord[:, 2:] += box_cord[:, :2]

    # calculate ldmk coordinate
    ldmk = ldmk_output[pre_select_boxes_index, :]
    ldmk[:, 0::2] = pre_select_anchor[:, 0:1] + ldmk[:, 0::2] * VARIANCE[0] * pre_select_anchor[:, 2:3]
    ldmk[:, 1::2] = pre_select_anchor[:, 1:2] + ldmk[:, 1::2] * VARIANCE[0] * pre_select_anchor[:, 3:4]

    # get prob
    box_prob = prob[pre_select_boxes_index, 1]

    return box_prob, box_cord, ldmk


def nms(box_prob, box_cord, ldmk):
    '''
    perform nms suppression
    :return:
    '''

    best_bboxes = []
    cls_bboxes = box_prob.copy()

    # loop over all boxes
    while cls_bboxes.shape[0] > 0:
        max_index = np.argmax(cls_bboxes)
        cls_bboxes = np.concatenate([cls_bboxes[: max_index], cls_bboxes[max_index + 1:]])

    pass


def decode_bbox(bbox, anchors=ANCHOR, variances=VARIANCE):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """

    boxes = np.concatenate((
        anchors[:, :2] + bbox[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float = NMS_THRES) -> np.ndarray:

    '''
    non maximum suppression by iou
    :param bbox:
    :param score:
    :param thresh:
    :return:
    '''

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def crop_faces(draw_img, bboxs, landms):

    face_imgs = []
    face_landmarks = []
    vaild_bboxs = []
    valid_img_index = []
    
    for i, (box, landm) in enumerate(zip(bboxs.astype(int), landms)):
      # crop face region
      cx, cy = (box[:2] + box[2:]) // 2
      halfw = np.max(box[2:] - box[:2]) // 2
      face_img: np.ndarray = draw_img[cy - halfw:cy + halfw, cx - halfw:cx +
                                      halfw]
      face_img_wh = face_img.shape[1::-1]
      if face_img_wh[0] == face_img_wh[1] and min(face_img_wh) > 10:
        face_landm = np.reshape(landm, (-1, 2)) - np.array(
            [cx - halfw, cy - halfw], 'int32')
        face_imgs.append(face_img)
        face_landmarks.append(face_landm)
        vaild_bboxs.append(box)
        valid_img_index.append(i)
            
    return valid_img_index, vaild_bboxs, face_imgs, face_landmarks


def face_algin_by_landmark(face_img: np.ndarray, face_landmark: np.ndarray,
                             template: np.ndarray=TEMPLATE) -> np.ndarray:
    img_dim = face_img.shape[:2][::-1]
    M, _ = cv2.estimateAffinePartial2D(face_landmark, img_dim * template)
    warped_img = cv2.warpAffine(face_img, M, img_dim)
    h_ratio = img_dim[0]
    w_ratio = int(h_ratio * FACE_DIMENSION[0] / FACE_DIMENSION[1])
    resized = cv2.resize(warped_img[:, int((h_ratio-w_ratio)/2):int((h_ratio+w_ratio)/2)], tuple(FACE_DIMENSION))
    return resized


def face_recognition(feature, database, threshold=0.5):
    feature_norm = feature / np.linalg.norm(feature, 2, -1, keepdims=True)  # normalization
    result = np.dot(database, feature_norm.T)
    return np.argmax(result), np.max(result)


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4]])
    b = np.array([[2, 3, 4, 5], [1, 2, 3, 4]])
    print(box_iou(a, b))
