# -*- coding: utf-8 -*- 
# @Time : 2024/2/27 10:55 
# @Author : LifeIsStrange24
# @File : image_fixed_table_area.py

from table_contours import *

from params import ROW_HEIGHT


def get_page_rows(binary,is_table=True):
    '''
    获取表格横线的纵坐标
    Args:
        binary: 黑白图
        is_table: 是否是表格，不是表格的话不用加最后一行

    Returns:行的y坐标

    '''

    dilated_col_z = dilate_line(binary, type='horizontal', x_scale=20)  # 20 chongqin  zhejiang获取行计算表 shanghai

    ys, xs = np.where(dilated_col_z > 0)
    point_arr = drop_duplicated_row_points(ys, max_span=ROW_HEIGHT)
    if is_table and binary.shape[0] - point_arr[-1] > ROW_HEIGHT:  # 最后一行无
        point_arr.append(binary.shape[0])
    # point_arr = drop_duplicated_points(ys)
    # logger.info('横线的y坐标,行：{},列表:{}'.format(len(point_arr) - 1, point_arr))
    return point_arr



def get_standard_table_image(gray, table):
    '''
    获取表格图片
    Args:
        gray:
        table:

    Returns:

    '''

    sorted_rect = get_sorted_rect(table)
    gray_z = perTran(gray, sorted_rect)
    binary_z = cv2.adaptiveThreshold(~gray_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, -5)

    return gray_z, binary_z

def get_muti_tables_images( gray, tables):
    '''
    获取多个table结果


    @param gray: 灰度图
    @param table:表格四个点坐标

    @return:返回解析结果
    '''
    gray_z_list = []
    binary_z_list = [] 
    for index,table in enumerate(tables):
        gray_z, binary_z = get_standard_table_image(gray, table)
        gray_z_list.append(gray_z)
        binary_z_list.append(binary_z)

    return gray_z_list,binary_z_list


def get_y_pos_by_vertical_line(binary, y_scale):
    '''

    Args:
        lsd:
        binary:
        y_scale:

    Returns:

    '''

    binary_dilate = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))) #膨胀 (10,3)改

    rows_z, cols_z = binary.shape
    size = (1, rows_z // y_scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)

    opening = cv2.morphologyEx(binary_dilate, cv2.MORPH_OPEN, kernel, 1)
    dlines = lsd.detect(opening)[0]

    pos = [[x[0][0], x[0][1]] for x in dlines] + [[x[0][2], x[0][3]] for x in dlines]
    top_pos_list = [x for x in pos if x[1] == min([x[1] for x in pos])]  # 最小的y
    top_pos = [x for x in top_pos_list if x[0] == min([x[0] for x in top_pos_list])][0]  # 最小的x

    bottom_pos_list = [x for x in pos if x[1] == max([x[1] for x in pos])]  # 最大的y
    bottom_pos = [x for x in bottom_pos_list if x[0] == min([x[0] for x in bottom_pos_list])][0]  # 最小的x

    x1, y1, x2, y2 = top_pos + bottom_pos

    return int(y1), int(y2), int(x1), int(x2)


def get_y_pos_by_line(rotated_binary):
    '''
    获取纵线的开始和结束点
    吉林 不能靠轮廓获取，因为有页码影响
    :param rotated_binary:
    :return:
    '''
   
    # 数字太少可能报错
    y_start, y_end, x_start, x_end = get_y_pos_by_vertical_line(
        rotated_binary[:, rotated_binary.shape[1] // 2 - 200:rotated_binary.shape[1] // 2 + 200], 20)  # 默认y_scale=5

    y_end += 60  # 重庆是80.
    y_start -= 50
    # x_start = max(x_start, x_end)
    # x_start = rotated_binary.shape[1] // 2 - 200 + x_start
    return y_start, y_end



def get_table_area(file,gray):
    '''

    :param file: 文件名称
    :param gray: 灰度图
    :param table_split_implicit:表格类型
    :param place_name: 地名
    :return:
    '''

    tables = get_table(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5)
    # 中间对半分

    right_tables = [x for x in tables if x[2][0] >= (gray.shape[1] * 0.4)]
    left_tables = [x for x in tables if x[2][0] < (gray.shape[1] * 0.4)]
    left_gray_z_list, _ = get_muti_tables_images(gray, left_tables)
    if right_tables:
        right_gray_z_list, _ = get_muti_tables_images(gray, right_tables)
        gray_z_list = left_gray_z_list + right_gray_z_list

    else:
        gray_z_list = left_gray_z_list

    return gray_z_list



