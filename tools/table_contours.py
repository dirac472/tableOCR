# -*- coding: utf-8 -*- 
# @Time : 2024/2/26 16:01 
# @Author : LifeIsStrange24
# @File : image_extract_table.py  提取表格四个顶点
from base_utils import get_sorted_rect,get_y_sorted_contours
import cv2
import numpy as np

def pre_process(gray):
    '''
    预防扫描图像有黑边
    Args:
        gray:

    Returns:

    '''
    gray_copy = gray.copy()
    gray_copy[:50, :] = 255
    gray_copy[-50:, :] = 255
    gray_copy[:, :50] = 255
    gray_copy[:, -50:] = 255

    return gray_copy


def get_table(gray, max_box_ratio=10, min_table_area=0):
    '''

    :param gray:
    :param max_box_ratio: 表格高宽比，超过比例将去除
    :param min_table_area:
    :return:
    '''
    error_black_edge = 0 # 是否存在黑边
    # 处理图像
    gray_copy = pre_process(gray)
    # gray_copy = pre_process(gray)
    canny = cv2.Canny(gray_copy, 200, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.dilate(canny, kernel)
    _, contours, HIERARCHY = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_LIST 有时候canny准 有时候binary更容易找到

    # draw_contours(gray, contours)

    if not min_table_area:
        min_table_area = gray.shape[0] * gray.shape[1] * 0.01   # 最小的矩形面积阈值
    candidate_table = [cnt for cnt in contours if cv2.contourArea(cnt) > min_table_area]  # 计算该轮廓的面积
    candidate_table = sorted(candidate_table, key=cv2.contourArea, reverse=True)
    # draw_contours(gray, candidate_table)
    # if not place_name.startswith('fujian'):
    candidate_table = get_y_sorted_contours(candidate_table) #  按照高度排序
    area_list = [cv2.contourArea(cnt) for cnt in candidate_table]

    if not candidate_table:
        return []

    table = []
    for i in range(len(candidate_table)):
        cnt = candidate_table[i]
        area = cv2.contourArea(cnt)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)  # boxPoints返回四个点顺序：右下→左下→左上→右上(实际上不定
        box = np.int0(box)
        # draw_contours(gray,[box])

        sorted_box = get_sorted_rect(box)  # 左上 右上 右下 左下
        box_ratio = (sorted_box[2][0] - sorted_box[3][0]) / (sorted_box[2][1] - sorted_box[1][1])
        # # logger.info('表格宽高比：{}'.format(box_ratio))
        if box_ratio > max_box_ratio or box_ratio < 1/max_box_ratio:  # 宽/高大于max_box_ratio
            continue
        result = [sorted_box[2], sorted_box[3], sorted_box[0], sorted_box[1]]  # 右下 左下 左上 右上

        result = [x.tolist() for x in result]
        table.append(result)


    if area_list and max(area_list) > gray.shape[0] * gray.shape[1] * 0.95: # 应该黑边情况
        print('该页可能存在黑边')
        error_black_edge = 1

    return table
