# -*- coding: utf-8 -*- 
# @Time : 2024/3/1 10:08 
# @Author : LifeIsStrange24
# @File : image_position.py

from base_utils import *




def repire_y_nums(binary_z, y_scale, right_num, max_try_times=50):
    '''
    根据制定的纵线数量选择参数 改进 可以用二分查找 或者
    @param binary_z:
    @param y_scale:
    @param right_num:正确纵线数量
    @param max_try_times:
    @return:
    '''
    y_point_arr = []
    num = 0
    while y_scale > 0 and y_scale <= 50:
        num += 1
        y_point_arr = get_page_cols_x_array(binary_z, y_scale, max_span=50)  # 大于一个单元格长度
        print('len(y_point_arr):',len(y_point_arr))
        if len(y_point_arr) < right_num:
            y_scale += 1
        elif len(y_point_arr) > right_num:
            y_scale -= 1
        else:
            break
        if y_scale < 0 or y_scale > 50 or num > max_try_times:
            print(5 / 0)
    return y_point_arr


def get_bitwise_and(binary_z, x_scale=10, y_scale=20,line_merge = False):
    '''
    返回表格交叉点
    Args:
        binary_z:
        x_scale:
        y_scale:

    Returns:

    '''

    dilated_row_z = dilate_line(binary_z, type='horizontal', x_scale=x_scale, y_scale=y_scale)
    dilated_col_z = dilate_line(binary_z, type='vertical', x_scale=x_scale, y_scale=y_scale)
    # self.cv_show(dilated_col_z, 'dilated_col_z')
    # 标识表格轮廓
    merge = cv2.add(dilated_row_z, dilated_col_z)  # 进行图片的加和
    # cv2.imshow("entire_excel_contour", merge)
    # cv2.waitKey()
    # self.cv_show(merge,'mearg')

    # 将识别出来的横竖线合起来 对二进制数据进行“与”操作
    bitwise_and_z = cv2.bitwise_and(dilated_col_z, dilated_row_z)
    # cv2.imshow("bitwise_and_z", bitwise_and_z)
    # cv2.waitKey()
    if line_merge:
        return merge
    else:
        return bitwise_and_z


def get_table_coordinate(bitwise_and, max_span=10):  # max_span=50
    '''
    从图片中获取表格横纵坐标
    '''
    # 将焦点标识取出来
    ys, xs = np.where(bitwise_and > 0)

    x_point_arr = drop_duplicated_points(xs, max_span)
    y_point_arr = drop_duplicated_points(ys, max_span)

    if bitwise_and.shape[0] - y_point_arr[-1] > 50:

        y_point_arr.append(bitwise_and.shape[0] - 1)
    if y_point_arr[0] > 50:  # 最后一列没有交叉点
        y_point_arr.insert(0, 0)
    if bitwise_and.shape[1] - x_point_arr[-1] > 50:
        x_point_arr.append(bitwise_and.shape[1] - 1)
    if x_point_arr[0] > 50:  # 最后一列没有交叉点
        x_point_arr.insert(0, 0)

    return x_point_arr, y_point_arr

def get_position_arr(binary_z, cols_list):
    '''
    通过二值图获取交叉点坐标

    Args:上海计算横纵坐标，其余省份计算行数
        binary_z:
        cols_list:

    Returns:

    '''
    right_num = len(cols_list) + 1
    bitwise_and_z = get_bitwise_and(binary_z)

    #  获取每个单元格位置dd
    x_point_arr, y_point_arr = get_table_coordinate(bitwise_and_z, max_span=25)  # 如果异常修改为10

    if len(x_point_arr) != right_num:
        x_point_arr = repire_y_nums(binary_z, y_scale=20, right_num=right_num)

    return x_point_arr, y_point_arr

def get_page_rows( binary, place_name, is_table=True,ROW_HEIGHT =60):
    '''
    获取表格横线的纵坐标
    Args:
        binary: 黑白图
        place_name:省份名称
        is_table: 是否是表格，不是表格的话不用加最后一行

    Returns:行的y坐标

    '''
    dilated_col_z = dilate_line(binary, type='horizontal', x_scale=20)
    ys, xs = np.where(dilated_col_z > 0)
    point_arr = drop_duplicated_row_points(ys, max_span=ROW_HEIGHT) #
    if is_table and binary.shape[0] - point_arr[-1] > ROW_HEIGHT:  # 最后一行无
        point_arr.append(binary.shape[0])

    return point_arr




def get_number_contours(cell, place_name, iterations=5, size=(3, 3)):
    '''
    通过腐蚀等操作获取文本轮廓 两个数字可能报错
    Args:
        cell: 图片
        place_name: 省份名称
        logger: 日志器
        iterations:膨胀迭代次数
        size:结构元素的大小

    Returns:数字轮廓

    '''
    number_height = 30
    if place_name == 'guangxi':
        number_height = 50
    element = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    canny = cv2.Canny(cell, 200, 255)
    dilation = cv2.dilate(canny, element, iterations=iterations)  # iterations=5 两个数字也能连在一起，paddle两个数字很容易识别
    #  v_show(dilation)
    # 此处使用RETR_LIST 而不是RETR_EXTERNAL 因为图像可能四面都有方框
    _, total_contours, HIERARCHY = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_LIST
    total_contours = sorted(total_contours, key=cv2.contourArea, reverse=True)  # 可能表格外别的文字轮廓
    # draw_contours(cell,total_contours)
    # 去掉边框线条
    # min_number_area = int((cell.shape[0]*cell.shape[1])/100)  # 单/双数数字最小面积 500 dialate=5
    max_number_area = int(cell.shape[0] * cell.shape[
        1] * 0.8)  # 5000 #int((cell.shape[0]*cell.shape[1])/10) int((cell.shape[0]*cell.shape[1])*0.8)
    # 计算该轮廓的面积   面积小的都筛选掉、按照效果自行设置
    area_list = [cv2.contourArea(cnt) for cnt in total_contours]

    contours = [cnt for cnt in total_contours if cv2.contourArea(cnt) < max_number_area]
    result_contours = []  # 数字应该靠近中间

    for i, cnt in enumerate(contours):

        box_width, box_height = cv2.minAreaRect(cnt)[1][0], cv2.minAreaRect(cnt)[1][1]  # 长宽
        # logger.info('第{}个轮廓 高度{} 宽度{}'.format(i, box_width, box_height))
        if cv2.minAreaRect(cnt)[2] > 45:
            box_width, box_height = box_height, box_width

        rect_x_center = cv2.minAreaRect(cnt)[0][0]  # 中心坐标
        rect_y_center = cv2.minAreaRect(cnt)[0][1]  # 中心坐标
        if len(contours) > 1 and box_width * box_height > max_number_area:  # 被包围的数字会被
            # logger.warn('第{}个轮廓box_height box_width,最小的矩阵面积超过阈值'.format(i))
            ...
        elif len(contours) > 1 and box_height > 0.9 * cell.shape[0] or box_height > 0.9 * cell.shape[1]:
            # logger.warn('第{}个轮廓box_height：{} box_width：{}长度超过高阈值{}或者宽阈值{}'
            #             .format(i, box_height, box_width, cell.shape[0], cell.shape[1]))
            ...
        elif not (0.1 < (rect_x_center / cell.shape[1]) < 0.9)  :  # 去掉小脏点 通过位置（不在中间的边界线）
            ...
        elif not (0.2 < (rect_y_center / cell.shape[0]) < 0.8) and place_name not in ['shanghai', 'shanghai_zk',
                                                                                      'guangxi', 'jilin']:
            # logger.warn('第{}个轮廓y坐标{},不在中心,不包括上海和广西和吉林 cell{}'.
            #             format(i, cv2.minAreaRect(cnt)[0], cell.shape))
            ...
        # elif box_height < cell.shape[1] * 0.25:
        #     # logger.warn('第{}个轮廓高度{}小于单元格{}高度4倍'.format(i, box_height, cell.shape[1]))
        elif box_height / 5 > box_width or box_width / 5 > box_height:  # 吉林列不止一个数字且没有纵线
            # logger.warn('第{}个轮廓高度 宽度比例超过5倍，判断为边界点，长：{} ，宽：{}'.
            #             format(i, box_width, box_height))
            ...
        elif box_height < number_height:
            # logger.warn('第{}个轮廓高度{}小于字体高度{}'.format(i, box_height, number_height))
            ...
        else:
            result_contours.append(cnt)

    contours = get_y_sorted_contours(result_contours)  # 左上 右上 右下 左下排列
    return contours

def get_padding_number_img(cell_child, zi_contours):
    '''
    通过轮廓获取数字图片， 并且加上padding ，如果不padding 3 可能识别为2,增加识别概率
    :param cell_child: img
    :param zi_contours: 轮廓
    :return:

    '''
    # 每个轮廓的box
    zi_box_list = [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) for cnt in zi_contours]
    zi_box_list = repair_negative_position(zi_box_list)  # 第一个像素点就是字体 轮廓为负

    #  外接矩形 四个点排序 按照左上 右上 右下 左下
    rect_child = [get_sorted_rect(rect) for rect in zi_box_list][0]

    cell_child = cell_child[ min([x[1] for x in rect_child]):max([x[1] for x in rect_child]),  # y
                 min([x[0] for x in rect_child]):max([x[0] for x in rect_child]),  # x
                 ]
    # 补成正方形
    padding_size = int(abs(cell_child.shape[0] - cell_child.shape[1]) / 2)
    top_size, bottom_size, left_size, right_size = (0, 0, padding_size, padding_size)
    padding_cell_child = cv2.copyMakeBorder(cell_child, top_size, bottom_size, left_size, right_size,
                                            borderType=cv2.BORDER_REPLICATE)

    return padding_cell_child

def get_y_subset_arr(cell, place_name, iterations=5, size=(3, 3)):
    '''
    有的表格多行之间没有横线
    Args:
        cell: 图片
        place_name:省份地名
        logger: 日志器
        iterations:获取数字轮廓时膨胀的次数
        size:获取数字轮廓时膨胀的结构元素的大小

    Returns:行的y坐标
    '''
    contours = get_number_contours(cell, place_name, iterations, size)  # draw_contours(cell,contours)
    # draw_contours(cell, contours)
    # 通过轮廓获取坐标
    box_list = [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) for cnt in contours]
    rects = [get_sorted_rect(rect) for rect in box_list]
    y_subset_point_arr = [list(x[0])[1] for x in rects]
    y_subset_point_arr = drop_duplicated_points(y_subset_point_arr, max_span=50)  # 比如两位数 会有4个轮廓
    y_subset_point_arr[0] = 0  # 设置y坐标开始位置为0

    y_subset_point_arr.append(cell.shape[0])
    y_subset_point_arr.sort()

    return y_subset_point_arr
