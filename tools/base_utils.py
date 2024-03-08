# -*- coding: utf-8 -*- 
# @Time : 2024/2/26 11:00 
# @Author : LifeIsStrange24
# @File : base_utils.py
import cv2
import numpy as np
import sys

from sqlalchemy import create_engine
import pymysql

from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
pymysql.install_as_MySQLdb()
engine = create_engine('mysql+mysqldb://root:root@localhost/ocr')

# 创建 Session 工厂类
Session = sessionmaker(bind=engine)
# 创建一个新的会话实例
session = Session()

lsd = cv2.createLineSegmentDetector(0, 1)


def get_eliminate_texted_binary(binary):
    '''
    消除文本：辽宁身份
    :param binary:
    :return:
    '''
    diate_rotated_binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10)))
    text_contours = get_text_contours_to_eliminate(diate_rotated_binary, iterations=1, size=(20, 3))
    # draw_contours(diate_rotated_binary, text_contours)
    cv2.drawContours(binary, text_contours, -1, (0, 0, 0), -1)  # 消除文本 保留直线
    return binary


def get_text_contours_to_eliminate(cell, iterations=1, size=(3, 3), flag=50, return_border_contours=False,
                                   min_area=300) -> object:
    '''
    高宽比大于flag的为表格线，消除，
    Args:
        cell:
        iterations:
        size:
        flag:
        return_border_contours:
        min_area: 多大面积以下为杂点

    Returns:

    '''
    element = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    canny = cv2.Canny(cell, 200, 255)

    dilation = cv2.dilate(canny, element, iterations=iterations)  # iterations=5 两个数字也能连在一起，paddle两个数字很容易识别

    _, total_contours, HIERARCHY = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw_contours(cell,total_contours)
    contours = [cnt for cnt in total_contours if cv2.contourArea(cnt) > min_area]  # 取出无关数字 括号600面积
    result_contours = []
    border_contours = []
    for i, cnt in enumerate(contours):
        box_width, box_height = cv2.minAreaRect(cnt)[1][0], cv2.minAreaRect(cnt)[1][1]  # 长宽

        if cv2.minAreaRect(cnt)[2] > 45:
            box_width, box_height = box_height, box_width

        if box_height / box_width > flag or box_width / box_height > flag:  # 纵线
            border_contours.append(cnt)
            # draw_contours(cell,[cnt])
        else:
            result_contours.append(cnt)
    if return_border_contours:
        return border_contours
    else:
        return result_contours


def get_y_arr_by_contours(cell, row_max_span, flag=5, number_height=0, size=(10, 3)):
    '''
    计算行的数量，还有每一行文字的开始与结尾纵坐标
    Args:
        cell:
        row_max_span:
        flag:    消除横线与纵线的倍数
        number_height: 低于number_height的算杂点，不计算
        size ： 一般字体，小一点且无杂质的字体设置(20,3)
    Returns:

    '''

    cell_test = cell.copy()
    # 对于一些省份需要消除横线
    # text_contour = get_text_contours_to_eliminate(cell_test, iterations=5, flag=flag, return_border_contours=True, min_area=0) # iterations1 改 5
    # draw_contours(cell_test, text_contour)
    # cv2.drawContours(cell_test, text_contour, -1, (255, 255), -1)

    # 对于一些省份 需要 消除小点，杂点
    contours = get_text_contours(cell_test, iterations=1, size=size, dot_area=500)  # 为什么设置为 是否可以设置成3
    # draw_contours(cell_test, contours)

    if number_height:
        result_contours = []  # draw_contours(cell,contours)
        for i, cnt in enumerate(contours):
            box_width, box_height = cv2.minAreaRect(cnt)[1][0], cv2.minAreaRect(cnt)[1][1]  # 长宽 为
            # 什么不选择bounding_rect?

            if cv2.minAreaRect(cnt)[2] > 45:
                box_width, box_height = box_height, box_width

            if box_height < number_height:  # 小杂点
                print('第{}个轮廓高度{}小于字体高/宽度 {}'.format(i, box_height, number_height))

            else:
                result_contours.append(cnt)
        contours = result_contours

    # draw_contours(cell_test, contours)
    #
    if len(contours) > 0:
        # 杂点去除干净了 【公】 可能只保留了一部分
        y_subset_arr = [cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in contours]
        y_subset_arr = drop_duplicated_points(y_subset_arr, max_span=row_max_span)  # 比如两位数 会有4个轮廓
        y_subset_arr.insert(0, 0)  # 设置y坐标开始位置为0 两行连在一起 没要没有完全连接也可以计算正确
    else:
        y_subset_arr = []
    return y_subset_arr


def get_text_contours(cell, iterations=5, size=(20, 3), number_height=0, dot_area=0):
    '''

    :param cell:
    :param iterations:
    :param size:
    :param number_height: 比数字高度还小的就是杂点
    :return:
    '''
    element = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    canny = cv2.Canny(cell, 200, 255)
    dilation = cv2.dilate(canny, element, iterations=iterations)  # 膨胀
    _, total_contours, HIERARCHY = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 此处使用RETR_LIST 而不是RETR_EXTERNAL 因为图像可能四面都有方框

    contours = sorted(total_contours, key=cv2.contourArea, reverse=True)
    # draw_contours(cell, contours)
    # result_contours = []
    # for i, cnt in enumerate(contours):
    #     # box_width, box_height = cv2.minAreaRect(cnt)[1][0], cv2.minAreaRect(cnt)[1][1]  # 长宽
    #     box_width, box_height = cv2.boundingRect(cnt)[2], cv2.boundingRect(cnt)[3]
    #     if box_height < number_height or box_width < number_height:  # 小杂点
    #         print('小杂点')
    #     else:
    #         result_contours.append(cnt)
    if number_height:
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[3] > number_height]  # 比数字高度小的就是杂点
    if dot_area:
        contours = [cnt for cnt in total_contours if cv2.contourArea(cnt) > dot_area]

    contours = get_y_sorted_contours(contours)  # 左上 右上 右下 左下排列

    return contours


def save_db(table_name,df):
    df.to_sql(table_name, con=engine, index=False, if_exists='append')



def get_angle_by_vertical_line( binary, y_scale = 20, file=''):
    '''
    通过lsd检测纵线，计算角度
    Args:
        lsd:
        binary:
        y_scale:
        file:

    Returns:

    '''
    # rows_z, cols_z = binary.shape
    # #if place=='shanxi':
    # size = (1,rows_z // y_scale)  # 根据线条长度调节参数
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    # eroded = cv2.erode(binary, kernel, iterations=1)  # 腐蚀
    # # cv_show(eroded)
    # ys, xs = np.where(eroded > 0)
    # sort_ys = sorted(list(set(xs)))
    # rows_list = drop_duplicated_row_points(sort_ys, max_span=50)  # 记录横线坐标
    # dlines = lsd.detect(eroded[:, rows_list[0] - 100:rows_list[0] + 100])[0]
    dilated_col_z = dilate_line(binary, type='vertical', y_scale=y_scale) # yscale太小可能找不到
    dlines = lsd.detect(dilated_col_z)[0]

        # print(5/0)
    pos = [[x[0][0], x[0][1]] for x in dlines] + [[x[0][2], x[0][3]] for x in dlines]
    top_pos_list = [x for x in pos if x[1] == min([x[1] for x in pos])]  # 最小的y
    top_pos = [x for x in top_pos_list if x[0] == min([x[0] for x in top_pos_list])][0]  # 最小的x

    bottom_pos_list = [x for x in pos if x[1] == max([x[1] for x in pos])]  # 最大的y
    bottom_pos = [x for x in bottom_pos_list if x[0] == min([x[0] for x in bottom_pos_list])][0]  # 最小的x

    x1, y1, x2, y2 = top_pos + bottom_pos

    angle =  cal_angle(x1, y1, x2, y2)
    if angle==0:
        print('%s 没有检测到直线' % file)
    return angle

def get_vertical_line_target_img(gray,line_index=1, span=0):
    '''
    纵线上方可能有字 jiangshu
    gray 图像
    line_index 第一条线就是1
    先canny
    纵向膨胀
    腐蚀文字和竖线
    获取横线纵坐标
    截图
    '''
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -5)

    canny = binary

    eroded = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150)), iterations=1)  # 腐蚀纵线 还有字

    # eroded = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (1, cols_z // k)), iterations=1)  # 腐蚀纵线 还有字
    ys, xs = np.where(eroded > 0)
    xs = list(set(xs))
    # 第一条直线开始和结束纵坐标
    sort_xs = sorted(list(set(xs)))
    target_img = eroded[:, sort_xs[0] - span:sort_xs[-1] + span]
    return target_img


def get_horizontal_line_target_img(gray, place='', line_index=1, span=0):
    '''

    line_index
    先canny
    纵向膨胀
    腐蚀文字和竖线
    获取横线纵坐标
    截图
    :param gray:
    :param place:
    :param line_index: line_index=1， 获取第一条线的上下span距离高度
    :param span: 获取线条的上下宽度
    :return: 返回 包含线条的二值图 图像
    '''
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -5)

    canny = binary
    # 计算第一条横线和最后一条横线的两端坐标
    rows_z, cols_z = binary.shape
    k = 20
    # 保留横线
    eroded = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (cols_z // k, 1)), iterations=1)  # 腐蚀纵线 还有字
    ys, xs = np.where(eroded > 0)
    rows_list = drop_duplicated_row_points(ys, max_span=50)  # 记录横线坐标

    # 第一条直线开始和结束纵坐标
    sort_ys = sorted(list(set(ys)))

    first_line_y_end = sort_ys[sort_ys.index(rows_list[line_index]) - 1]
    first_line_y_start = rows_list[line_index - 1]
    target_img = eroded[max(first_line_y_start - span, 0):first_line_y_end + span, :]
    return target_img


def get_line_position(eroded, detection='horizontal'):
    '''
    获取线条的坐标
    :param eroded:
    :param detection: 检测方向
    :return:
    '''
    dlines = lsd.detect(eroded)[0]
    pos = [[x[0][0], x[0][1]] for x in dlines] + [[x[0][2], x[0][3]] for x in dlines]
    if detection == 'horizontal':

        top_pos_list = [x for x in pos if x[0] == min([x[0] for x in pos])]  # 最小的x
        top_pos = [x for x in top_pos_list if x[1] == min([x[1] for x in top_pos_list])][0]  # 最小的y

        bottom_pos_list = [x for x in pos if x[0] == max([x[0] for x in pos])]  # 最大的x
        bottom_pos = [x for x in bottom_pos_list if x[1] == max([x[1] for x in bottom_pos_list])][0]  # 最大的y
    else:
        top_pos_list = [x for x in pos if x[1] == min([x[1] for x in pos])]  # 最小的y
        top_pos = [x for x in top_pos_list if x[0] == min([x[0] for x in top_pos_list])][0]  # 最小的x

        bottom_pos_list = [x for x in pos if x[1] == max([x[1] for x in pos])]  # 最大的y
        bottom_pos = [x for x in bottom_pos_list if x[0] == min([x[0] for x in bottom_pos_list])][0]  # 最小的x
    x1, y1, x2, y2 = top_pos + bottom_pos

    return x1, y1, x2, y2




def get_line_x_start(dilated_row, black_edge, is_left=True):
    '''
    获取 dilated_row 线条起点
    Args:
        dilated_row:
        black_edge:
        is_left:

    Returns:

    '''
    ys, xs = np.where(dilated_row > 0)
    x_start = min(xs) + black_edge
    if not is_left:
        x_start = min(xs)
    return x_start


def get_rows_list(dilated_row, black_edge):
    '''
    获取dilated_row 的列纵坐标
    Args:
        dilated_row:
        black_edge:

    Returns:

    '''
    ys, xs = np.where(dilated_row > 0)
    rows_list = drop_duplicated_row_points(ys, max_span=50)  # 记录横线坐标
    rows_list = [x + black_edge for x in rows_list]
    return rows_list

def get_page_cols_x_array(binary, y_scale, max_span, require_supplement=True):
    '''
    获取表格纵线横坐标
    Args:
        binary:
        y_scale:
        max_span:
        require_supplement:

    Returns:

    '''
    dilated_col_z = dilate_line(binary, type='vertical', y_scale=y_scale)
    ys, xs = np.where(dilated_col_z > 0)
    point_arr = drop_duplicated_row_points(xs, max_span)
    if require_supplement == True:
        if point_arr[0] > 50:
            # logger.info('第一条纵线纵坐标大于50，添加横线')
            point_arr.insert(0, 0)
        if binary.shape[1] - point_arr[-1] > 50:
            # logger.info('最后一条纵线纵坐标大于50，添加横线')
            point_arr.append(binary.shape[1])

    return point_arr



def get_page_rows_y_array(binary, row_height, x_scale=20):
    '''
    保留纵线，获取该图像行数
    cv2.imwrite('dilated_col_z.jpg', dilated_col_z)
    Args:
        binary:
        row_height:
        x_scale:

    Returns:

    '''

    dilated_col_z = dilate_line(binary, type='horizontal', x_scale=x_scale)
    ys, xs = np.where(dilated_col_z > 0)
    point_arr = drop_duplicated_row_points(ys, max_span=row_height)
    if binary.shape[0] - point_arr[-1] > row_height:
        point_arr.append(binary.shape[0])
    return point_arr

def get_y_sorted_contours(contours, only_rect=False):
    '''
    获取根据纵坐标排序后的轮廓点
    contours:很多轮廓点
    @return:轮廓点
    '''

    box_list = [np.array([
        [cv2.boundingRect(cnt)[0], cv2.boundingRect(cnt)[1]],
        [cv2.boundingRect(cnt)[0]+cv2.boundingRect(cnt)[-2], cv2.boundingRect(cnt)[1]],
        [cv2.boundingRect(cnt)[0]+cv2.boundingRect(cnt)[-2], cv2.boundingRect(cnt)[1]+cv2.boundingRect(cnt)[-1]],
        [cv2.boundingRect(cnt)[0], cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[-1]]
    ]) for cnt in contours]
    box_list.sort(key=lambda x: x[0][1])
    return box_list

def repair_negative_position(box_list):
    '''
    坐标值为负数，修正为正数
    @param box_list:
    @return:
    '''
    box_array = np.array(box_list)
    box_array[box_array < 0] = 0
    box_list = box_array.tolist()
    box_list = [np.array(x) for x in box_list]
    return box_list

def get_y_sorted_contours(contours, only_rect=False):
    '''
    获取根据纵坐标排序后的轮廓点(注意contours 每个点的坐标没有发生改变，box_list没有变动）
    @return:轮廓点
    '''
    # 每个轮廓的box
    box_list = [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) for cnt in contours]
    if [x for x in box_list if x.any() < 0]:  # 坐标是否有小于0的值
        print('坐标值为负数')
        box_list = repair_negative_position(box_list)

    # 根据y坐标排序
    if only_rect:
        rects = [get_sorted_rect(rect) for rect in box_list if len(rect) <= 4]
    else:
        rects = [get_sorted_rect(rect) for rect in box_list]
    y_subset_point_arr = [x[0][1] for x in rects]  # 按照纵坐标排序

    contours_y_sorted = [[box_list[i], y_subset_point_arr[i]] for i in range(len(box_list))]
    contours_y_sorted.sort(key=(lambda x: x[1]))

    contours = [x[0] for x in contours_y_sorted]
    return contours

def cal_angle(x1, y1, x2, y2, is_vertical=True):
    if x2 - x1 == 0:
        print("直线是竖直的")

        result = 90
    elif y2 - y1 == 0:
        print("直线是水平的")
        result = 0
    else:
        # 计算斜率
        k = -(y2 - y1) / (x2 - x1)
        # 求反正切，再将得到的弧度转换为度
        result = np.arctan(k) * 57.29577  # 逆时针

    if is_vertical:
        if result < 0:
            result += 90
        elif result == 90:
            result = 0
        else:
            result -= 90
        print("通过竖线计算得到直线倾斜角度为:{} 度".format(result))
    else:
        print("通过横线计算得到直线倾斜角度为:{} 度".format(result))
    result = round(result, 3)

    return result


def clean_gray(gray, ksize=3, difference=50):
    '''
    中值滤波清洗图片
    Args:
        gray:
        ksize:
        difference: 与255值（白色）差值

    Returns:

    '''
    gray_copy = gray.copy()
    gray_copy[(gray_copy >= (255 - difference))] = 255
    gray_copy = cv2.medianBlur(gray_copy, ksize)
    # 双边滤波 效果差不多 a2 = cv2.bilateralFilter(img, d=0, sigmaColor=10, sigmaSpace=10)
    return gray_copy


def get_median_blur_cell(cell, size=3):
    '''
    滤波处理
    :param cell:
    :param size:
    :return:
    '''
    median_blur_cell_child = cell.copy()
    median_blur_cell_child[(median_blur_cell_child >= 155)] = 255
    median_blur_cell_child = cv2.medianBlur(median_blur_cell_child, size)  # 有杂质会报错 listout of range
    return median_blur_cell_child


import os
import psutil

# 获取当前进程ID
pid = os.getpid()


def print_program_memory(pid):
    '''
    通过pid检查内存
    :param pid:
    :return:
    '''
    # 创建Process对象
    process = psutil.Process(pid)

    # 获取内存信息
    # mem_info = process.memory_info()
    # print(f"当前进程占用内存（RSS）: {mem_info.rss / 1024 ** 2:.2f} MB")
    # print(f"当前进程虚拟内存（VMS）: {mem_info.vms / 1024 ** 2:.2f} MB")

    # 或者使用更加简洁的方式直接获取 Resident Set Size (RSS)
    print(f"当前进程占用内存（RSS简化版）: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def check_table_complete(img_folder, select_df):
    '''
    检查是否所有文件都成功获取表格图像
    :param img_folder:
    :param select_df:
    :return:
    '''
    orginal_files = os.listdir(img_folder)
    ocr_files = [os.path.basename(file) for file in list(select_df['原始路径'])]
    print('没有获取到表格图像的文件：', set(orginal_files) - set(ocr_files))


def find_outlier_by_IQR(data):
    # 计算第一四分位数（Q1）、第三四分位数（Q3）和 IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    IQR = 50
    # 定义异常值阈值
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 检测异常值
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    if len(outliers) > 0:
        print("IQR潜在异常值：", outliers)


def check_table_shape(width_data, outlier_span=50):
    '''
    检查表格宽度，如果有黑线会被检查出来

    :param width_data:
    :param outlier_span: 与中位数相差表格宽度差多少为异常值
    :return:
    '''
    # find_outlier_by_zscore(width_data,outlier_span)
    # find_outlier_by_LOF(width_data, percentile=1)
    find_outlier_by_IQR(width_data)

def dilate_line(binary, type='vertical', x_scale=10, y_scale=5):
    '''
    获取竖线/横线腐蚀后的二值黑白图
    Args:
        binary:
        type:
        x_scale:
        y_scale:

    Returns:

    '''
    rows_z, cols_z = binary.shape


    # 识别横线:
    if type == 'horizontal':
        size = (cols_z // x_scale, 1)
    else:  # 'vertical'
        size = (1, rows_z // y_scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)

    eroded = cv2.erode(binary, kernel, iterations=1)  # 腐蚀
    dilated = cv2.dilate(eroded, kernel, iterations=1)  # 膨胀
    # cv2.imshow("excel_vline", dilated)
    # cv2.waitKey()  dilated = cv2.dilate( cv2.dilate(cv2.erode(binary, kernel, iterations=1), kernel, iterations=1), kernel, iterations=1)
    return dilated


def get_page_cols_x_array(binary, y_scale, max_span, require_supplement=True):
    '''
    获取表格纵线横坐标
    Args:
        binary:
        y_scale:
        max_span:
        require_supplement:

    Returns:

    '''
    dilated_col_z = dilate_line(binary, type='vertical', y_scale=y_scale)
    ys, xs = np.where(dilated_col_z > 0)
    point_arr = drop_duplicated_row_points(xs, max_span)
    if require_supplement == True:
        if point_arr[0] > 50:
            # logger.info('第一条纵线纵坐标大于50，添加横线')
            point_arr.insert(0, 0)
        if binary.shape[1] - point_arr[-1] > 50:
            # logger.info('最后一条纵线纵坐标大于50，添加横线')
            point_arr.append(binary.shape[1])

    return point_arr

def rotate_image( image, angle, if_fill_white = False):
    '''
    顺时针旋转
    Args:
        image:
        angle:
        if_fill_white:

    Returns:

    '''
    # dividing height and width by 2 to get the center of the image
    height, width = image.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width / 2, height / 2)

    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

    # rotate the image using cv2.warpAffine
    if not if_fill_white:
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height) )
    else:
        color = (255, 255) if len(image.shape)==2 else (255, 255,255)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=color)
    return rotated_image


def get_contours_x(table_contours):
    '''
    获取轮廓最大和最小x
    Args:
        table_contours:

    Returns:

    '''
    x_start_list = [ cv2.boundingRect(cnt)[0] for cnt in table_contours]
    x_end_list = [ cv2.boundingRect(cnt)[0]+cv2.boundingRect(cnt)[-2] for cnt in table_contours]
    table_min_x = min(x_start_list)
    table_max_x = max(x_end_list)
    return table_min_x, table_max_x

def drop_duplicated_row_points(pos, max_span, add_last_point = False):
    '''
    坐标点会重复
    Args:
        pos:
        max_span:
        add_last_point:是否加上最后一个点
    Returns:
    '''

    sort_point = np.sort(list(set(pos)))
    point_arr = [sort_point[0]]  # 每种类型数据max_span行都不一样
    for i in range(1, len(sort_point) - 1):
        # print(i, sort_point[i], sort_point[i] - point_arr[-1] )
        if sort_point[i] - point_arr[-1] > max_span:
            point_arr.append(sort_point[i])
    if add_last_point and (sort_point[-1] - point_arr[-1] > max_span):
        point_arr.append(sort_point[-1])
    return point_arr  # 最后一个点无法加入


def drop_duplicated_points(pos, max_span=10):
    '''
    获取去重后的坐标点，如果一个文字拆成两半，一半的文字又大于max_span,则行的纵坐标可能统计错误
    [4,5, 6, 7, 8,9,10, 1748, 1749, 1750, 1751,1762, 1763, 1764, 1765, 1766, 3504, 3505, 3506, 3507, 3508, 3509]
    Args:
        sort_point:
        max_span:

    Returns: [10, 1766, 3509]

    '''
    sort_point = np.sort(list(set(pos)))
    point_arr = []  # 存在一行分成两行误差
    for i in range(len(sort_point) - 1):
        if (sort_point[i + 1] - sort_point[i] > max_span):  #
            # if (sort_point[i + 1] - sort_point[i] > max_span):  #
            point_arr.append(sort_point[i])
    point_arr.append(sort_point[-1])

    return point_arr


def get_sorted_rect(rect):
    '''
    获取矩阵排序的四个坐标,方便透视变换使用
    rect包含坐标点为负数时，left_rect包含三个 right_rect包含1个坐标点，或者反之
    @param rect:
    @return:按照左上 右上 右下 左下排列返回

    '''
    try:
        mid_x = (max([x[1] for x in rect]) - min([x[1] for x in rect])) * 0.5 + min([x[1] for x in rect])  # 中间点坐标
        left_rect = [x for x in rect if x[1] < mid_x]
        left_rect.sort(key=lambda x: (x[0], x[1]))
        right_rect = [x for x in rect if x[1] > mid_x]
        right_rect.sort(key=lambda x: (x[0], x[1]))
        sorted_rect = left_rect[0], left_rect[1], right_rect[1], right_rect[0]  # 左上 右上 右下 左下
    except:
        # np.array_equal(order_points(rect), sorted_rect):
        sorted_rect = order_points(rect)

    return sorted_rect

def order_points(pts):
    '''
    坐标排序
    Args:
        pts:

    Returns:

    '''
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="int64")

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 获取右上角和左下角坐标点
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect[0], rect[1], rect[2], rect[3]



def perTran(image, rect):
    '''
    做透视变换
    image 图像
    rect  四个顶点位置:左上 右上 右下 左下
    '''

    tl, tr, br, bl = rect  # 左下 右下  左上 右上 || topleft topright 左上 右上 右下 左下
    # 计算宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 计算高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 定义变换后新图像的尺寸
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype='float32')
    # 变换矩阵
    rect = np.array(rect, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped




def cv_show(img):
    '''
    windows服务器上展示图片
    @param img:
    @param name:
    @return:
    '''
    if sys.platform != 'linux':
        # cv2.namedWindow('name', cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        # cv2.imshow('name', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        h, w = img.shape[:2]
        if max(h, w)>800:
            scale = 800 / max(h, w)
        else:
            scale = 1
        cv2.namedWindow('name', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('name', img)
        cv2.resizeWindow('name', int(w * scale), int(h * scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('linux无法show')

def draw_contours(img, contours, color=(255, 0, 0)):
    '''
    使用蓝色画出轮廓
    @param img:
    @return:
    '''
    test_img = img.copy()
    test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)  # 灰度图转彩色方便观看
    test_img_res = cv2.drawContours(test_img, contours, -1, color, 2)  # # -1是全部轮廓 1是第一个图像外圈,...最后一个参数是线条长度
    # test_img_res = cv2.drawContours(test_img, contours, -1, (0, 0, 0),
    #                                 2)

    cv2.imwrite('test_img_res.jpg', test_img_res)
    # cv2.imwrite('img_res.jpg', img)
    # test_img_res = cv2.drawContours(test_img, contours, -1, (0, 0, 0),
    #                                 2)  #
    if sys.platform != 'linux':
        cv_show(test_img_res)
    return test_img_res