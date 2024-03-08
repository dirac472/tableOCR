# -*- coding: utf-8 -*-
# @Time : 2024/2/26 16:18
# @Author : LifeIsStrange24
# @File :  main.py 主函数

import json

from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('./tools')  # 加上这一句，否则报错tools下的包找不到
# from tools import line_position 相对导入
from tools.line_position import get_position_arr,get_number_contours,get_padding_number_img
from tools.table_area import get_table_area,get_page_rows
from tools.base_utils import  *
from tools.image_ocr import  sort_paddle_result




def save_gray_z_list(imagePath, gray_z_list, table_image_folder):
    '''
    存在目录下
    :param gray_z_list:
    :return:
    '''
    table_image_width_list = []
    table_image_height_list = []
    table_image_path_list = []
    for i, gray_z in enumerate(gray_z_list):
        table_image_path = os.path.basename(imagePath)[:-4] + '_' + str(i) + '.jpg'
        save_path = os.path.join(table_image_folder, table_image_path).replace('\\', '/')
        table_image_path_list.append(save_path)
        table_image_height_list.append(gray_z.shape[0])
        table_image_width_list.append(gray_z.shape[1])
        cv2.imwrite(save_path, gray_z)
    res = pd.DataFrame(table_image_path_list, columns=['保存表格图像路径'])

    res['图像高度'] = table_image_height_list
    res['图像宽度'] = table_image_width_list
    res['原始路径'] = imagePath.replace('\\', '/')
    res['原始文件夹'] = imagePath.replace('\\', '/').split('/')[-2]
    res['省份'] = 'test'


    save_db(table_name, res)
    print('添加%s成功' % os.path.basename(imagePath))



def generate_gray_z_files(img_folder, table_image_folder):
    '''
    生成标准表格文件
    :param target_files_path_list:
    :param table_image_folder:
    :param table_split_implicit:
    :return:
    '''
    target_files_list = os.listdir(img_folder)
    target_files_path_list = [os.path.join(img_folder, x) for x in target_files_list]

    for imagePath in tqdm(target_files_path_list):
        print(imagePath)
        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_z_list = get_table_area(os.path.basename(imagePath), gray)
        # cv_show(gray_z_list[0]) 查看表格获取结果
        save_gray_z_list(imagePath, gray_z_list, table_image_folder)
        del img
        del gray
        del gray_z_list



def get_table_cell_text(gray, x_point_arr, y_point_arr, cols_list):
    '''
    @param gray:
    @param x_point_arr:
    @param y_point_arr:
    @param img_folder: 图片保存的文件夹
    np.array(y_point_arr[1:]) - np.array(y_subset_end_point_arr[:-1])
    @return:
    '''
    data = []
    start_index = header_row
    for i in range(start_index, len(y_point_arr) - 1):  # 不识别表头
        print('正在处理第%s行'%i)
        # 判断该行是学校还是专业信息
        text_list = []
        for j in range(len(x_point_arr) - 1):
                cell = gray[ y_point_arr[i] : y_point_arr[i+1], x_point_arr[j]:x_point_arr[j + 1] ]
                if j in single_number_index_list: # 脏点多的时候 1可能识别为11
                    # zi_contours =  get_number_contours( get_median_blur_cell(cell), iterations=5)  #
                    # padding_cell_child = get_padding_number_img(cell, zi_contours)
                    ocr_result = sort_paddle_result(cell, det=False)
                    text = ''.join([x[0] for x in ocr_result])
                else:
                    text = sort_paddle_result(cell)
                text_list.append(text)
        data.append(text_list)

    res = pd.DataFrame(data, columns=cols_list)
    res['table_height'] = y_point_arr[-1] - y_point_arr[0]
    return res

def get_table_position(gray_z):
    '''

    :param gray_z:
    :param place_name:
    :return:
    '''

    binary_z = cv2.adaptiveThreshold(~gray_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5)

    y_point_arr = get_page_rows(binary_z)

    # 通过表头数量，即纵线数量判断
    x_point_arr, _ = get_position_arr(binary_z, cols_list)  # 有些表格没有纵线，比如浙江

    return [x_point_arr, y_point_arr[header_row:], cols_list]



def get_filter_line_img(gray_z, binary_z, y_scale=20, x_scale=10,size=(3,3),iterations=2):
    '''
    获取过滤框线之后的图像,避免框线影响切割后的图片识别率
    Args:
        gray_z:
        binary_z:
        y_scale:
        x_scale:
        size:
        iterations:

    Returns:

    '''
    dilated_row_z = dilate_line(binary_z, type='horizontal', x_scale=x_scale)
    dilated_col_z = dilate_line(binary_z, type='vertical', y_scale=y_scale)  # 设置为50
    merge = cv2.add(dilated_row_z, dilated_col_z)  # 黑底白框
    # cv_show(merge)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,size)  # 膨胀
    # eroded = cv2.erode(merge, kernel, iterations=3)
    dilated = cv2.dilate(merge, kernel, iterations=iterations)
    # cv_show(dilated)
    filter_line_gray_z = cv2.add(gray_z, dilated)  # 去掉框线之后
    #    cv_show(filter_line_gray_z)
    return filter_line_gray_z



def preprocess_files( img_folder):
    '''

    :param place_name: 省份类型
    :param img_folder: 图像数据源
    :return:
    '''
    if not os.path.exists(img_folder):
        print('文件夹 %s 不存在' % img_folder)
    else:
        if not os.path.exists(table_image_folder):
            os.mkdir(table_image_folder)

        generate_gray_z_files(img_folder, table_image_folder)
        target_files_list = os.listdir(table_image_folder)
        target_files_path_list = [os.path.join(table_image_folder, x).replace('\\', '/') for x in target_files_list]
        target_files_path_list.sort()  # 排序
        file_table_index = 0 # 该页第几张图像
        for imagePath in tqdm(target_files_path_list):
            print(imagePath)
            img = cv2.imread(imagePath)
            gray_z = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary_z = cv2.adaptiveThreshold(~gray_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5)
            x_point_arr, y_point_arr, cols_list = get_table_position(gray_z)
            save_gray_z_position(imagePath, x_point_arr, y_point_arr)
            gray_z = get_median_blur_cell(gray_z)  # 滤波过滤下脏点，如果图片线条本身不清洗就去除这一步
            filter_gray_z = get_filter_line_img(gray_z, binary_z)
            res = get_table_cell_text(filter_gray_z, x_point_arr, y_point_arr, cols_list)
            filename = os.path.basename(imagePath)
            res['file_table_index'] = filename[:-4].split('_')[-1]
            res['imagePath'] = imagePath
            res.to_sql(result_table_name, con=engine, index=False, if_exists='append')
            res.to_excel(result_file,index=False)


def update_db_data(session, update_query, params):
    '''
    更新数据库
    :param update_query:
    :param params:
    :return:
    '''
    result = session.execute(update_query, params)

    try:
        session.commit()
    except Exception as e:
        print(f"更新操作失败: {e}")
        session.rollback()  # 如果出现异常，回滚事务
    else:
        affected_rows = result.rowcount
        if affected_rows > 0:
            print(f"更新操作成功，受影响的行数为：{affected_rows}")
    finally:
        session.close()


def save_gray_z_position(imagePath, x_point_arr, y_point_arr):
    json_str = json.dumps({'x_point_arr': [int(x) for x in x_point_arr],
                           'y_point_arr': [int(x) for x in y_point_arr]})

    row_span_list = np.array(y_point_arr[1:]) - np.array(y_point_arr[:-1])
    cols_span_list = np.array(x_point_arr[1:]) - np.array(x_point_arr[:-1])
    rows_num = len(row_span_list)
    cols_num = len(x_point_arr) - 1

    # 编写并执行更新语句

    update_query = text("UPDATE %s SET  `坐标` = :position, `列数` = :cols_num, `行数` = :rows_num, `行最大高度` = :max_row_height,`行最小高度` = :min_row_height  WHERE `保存表格图像路径` = :path"%table_name)
    params = {'position': json_str,
              'cols_num': cols_num,
              'rows_num': rows_num,
              'max_row_height': max(row_span_list),
              'min_row_height': min(row_span_list),
              'path': imagePath}

    update_db_data(session, update_query, params)
    print('添加%s成功' % os.path.basename(imagePath))



if __name__ == '__main__':
    # 两个数据库的表
    table_name = 'gray_z_info'# 保存表格的高，宽，位置信息
    result_table_name = 'gray_z_ocr_result' # 保存识别内容的db_name

    #表格的常数设置
    header_row = 1  # 表头有几行
    cols_list = ['a','b','c','d','e']  # 设置表头
    single_number_index_list = [3] # 某列可能多为单个数字，需要单独识别

    # 文件夹设置
    img_folder = './image'    # 表格图像的文件夹 表格需要同一个类型
    table_image_folder = './table_image'  # 保存标准的表格图像
    result_file = './result/test.xlsx'

    preprocess_files(img_folder)
