# -*- coding: utf-8 -*-
# @Time : 2024/2/26 10:34
# @Author : lxy
# @File : image_ocr.py
import re

from cnradical import Radical, RunOption
radical = Radical(RunOption.Radical)  # 获取偏旁

import paddle

from paddleocr import PaddleOCR
print('device', paddle.device.get_device())
ocr = PaddleOCR(lang="ch", show_log=False)



from base_utils import *

def is_chinese_char(char):
    '''
    判断字符串是否是中文
    :param char:
    :return:
    '''
    pattern = r'[\u4e00-\u9fff]'
    match_result = re.search(pattern, char)
    if match_result:
        return True
    else:
        return False

def del_ocr_repeat_words(ocr_result):
    '''
    通过识别坐标交叉，删除重复识别的数据
    Args:
        ocr_result:
    Returns:
        删除后的重复数据
    '''
    ocr_result = [[x[0], [x[1][0].replace(' ', ''), x[1][1]]] for x in ocr_result] #不要空格
    ocr_result = [x for x in ocr_result if x[1][0] != '']  # 可能识别出空字符串

    text_start_x_list = [x[0][0][0] for x in ocr_result]
    text_end_x_list = [x[0][1][0] for x in ocr_result]

    for i, text_end_x in enumerate(text_end_x_list[:-1]):
        flag = 0
        if text_start_x_list[i + 1] < text_end_x:
            if ocr_result[i][1][0] == '' or ocr_result[i + 1][1][0] == '': # 这个字符串是空 或者下个字符串是空
                break

            if radical.trans_ch(ocr_result[i + 1][1][0][0]) == ocr_result[i][1][0][-1]:#判断一个字是否是另一个字的偏旁
                print('汉字多识别出了偏旁，发生交叉，back前 %s ' % (' '.join([x[1][0] for x in ocr_result])))
                ocr_result[i][1][0] = ocr_result[i][1][0][:-1]
                print('汉字多识别出了偏旁，发生交叉，back后 %s ' % (' '.join([x[1][0] for x in ocr_result])))
                if ocr_result[i + 1][1][0] == '':#循环到最后一个文本内容就不要再循环了
                    break

            if ocr_result[i + 1][1][0][0] == ocr_result[i][1][0][-1]:#识别出相同的两个字
                print('发生交叉，back前 %s ' % (' '.join([x[1][0] for x in ocr_result])))
                ocr_result[i + 1][1][0] = ocr_result[i + 1][1][0][1:]
                print('发生交叉，back后 %s ' % (' '.join([x[1][0] for x in ocr_result])))
                if ocr_result[i + 1][1][0] == '':
                    break

            # 数字交叉
            if ((ocr_result[i + 1][1][0][0] == '3' and ocr_result[i][1][0][-1] == '8') or  (ocr_result[i + 1][1][0][0] == '6' and ocr_result[i][1][0][-1] == '5')):
                ocr_result[i + 1][1][0] = ocr_result[i + 1][1][0][1:]
                if ocr_result[i + 1][1][0] == '':
                    break

            if is_chinese_char(ocr_result[i + 1][1][0][0]) and is_chinese_char(ocr_result[i][1][0][-1]) and len(ocr_result[i][1][0]) > 1 and len(ocr_result[i + 1][1][0]) > 1:
                print('所有汉字交叉的数据 %s %s' % (ocr_result[i + 1][1][0][0], ocr_result[i][1][0][-1]))  # 查看交叉的数据是否处理
                ocr_result[i][1][0] = ocr_result[i][1][0][:-1]  # 就算是中文 三、四学年被识别成   一、四
                if ocr_result[i + 1][1][0] == '' or ocr_result[i][1][0] == '':
                    break
    return ocr_result



def sort_paddle_result_by_pos(result, val=30):
    '''
    对paddle返回的结果排序，纵坐标相差为val的默认为一行
    Args:
        result:paddle ocr结果
        val: 差值多少为一行

    Returns:

    '''
    if len(result)>0:
        result.sort(key=lambda x: x[0][0][1])  # y排序
        first_dot = result[0][0][0]
        other_dots = [x[0][0] for x in result[1:]]
        sort_dots = [[first_dot]]  # 第一个点  在第一行
        index = 0
        for dot in other_dots:
            if sort_dots[index] == [] or abs(dot[1] - np.mean([x[1] for x in sort_dots[index]])) < val:  # 同一行
                sort_dots[index].append(dot)
            else:  # 第二行
                index += 1
                sort_dots.append([dot])
        # print('共有%s行' % len(sort_dots))
        sort_result = []
        for dot in sort_dots:  # [item for sublist in sort_dots for item in sublist]:
            # ocr_result = [x for x in result if x[0][0] == dot]
            ocr_result = [x for x in result if x[0][0] in dot]
            sort_result.append(ocr_result)

        for sublist in sort_result:
            sublist.sort()
        sort_result = [item for sublist in sort_result for item in sublist]  # 返回原始格式
        # print(sort_result)
        return len(sort_dots), sort_result
    else:
        return 0, []

def get_nums_by_ocr_results(ocr_result,row_max_span):
    '''
    通过识别结果反推行数
    :param ocr_result:
    :param place_name:
    :param row_max_span:
    :return:
    '''
    # 设置行间距值，计算行数量是会用上

    # ocr结果识别行数
    num = 1 if (max([x[0][0][1] for x in ocr_result]) - min(
        [x[0][0][1] for x in ocr_result])) < row_max_span else 2
    return num




def check_ocr_complete(cell, ocr_result, number_height):
    '''
    通过轮廓检查文字是否前半部分没有识别出来，
    Args:
        cell:
        ocr_result:

    Returns:

    '''
    complete_flag = 1
    row_contours = get_text_contours(cell, iterations=5, size=(10, 3), number_height= number_height)
    if len(row_contours) > 0:
        contours_min_x = min([cv2.boundingRect(cnt)[0] for cnt in row_contours])
        min_x = min([x[0][0][0] for x in ocr_result])
        if min_x > contours_min_x + 100:
            complete_flag = 0
            # draw_contours(cell,row_contours)

    return complete_flag

def sort_paddle_result(cell,place_name='',number_height = 0, det=True):  # padding
    '''
    只有一行 按照x排序
    多行 先按照y 再按照x 排序
    Args:
        cell:

    Returns:

    '''
    row_rec_fail = -1 # 第几行识别失败了
    error_2rec1 = '' # 可能是两行识别成一行

    size = (10, 3)
    ratio = cell.shape[1] // cell.shape[0]
    row_max_span = 50 # 每一行的文字高度不超过50

    # 判断是否存在文本
    contours = get_text_contours(cell, iterations=5, size=(20, 3))
    if len(contours) == 0:
        return ''


    y_subset_arr = get_y_arr_by_contours(cell, row_max_span, flag=10, number_height=number_height,size=size)
    row_num = len(y_subset_arr) - 1

    info = ''
    if det == False:
        for i in range(len(y_subset_arr) - 1):  # 如果改行有两行数据，会识别成乱码
            row_cell = cell[y_subset_arr[i]:y_subset_arr[i + 1], :]
            ocr_row_result = ocr.ocr(row_cell, det=False)
            if ocr_row_result:
                row_str = ''.join([x[0] for x in ocr_row_result])
                info += row_str
            else:
                info = '识别异常'
        return info

    else:

        # 基本只有一行
        if ratio >= 10 :  #
            pad_cell = np.pad(cell, ((10, 10), (0, 0)), 'constant', constant_values=(255))
            ocr_result = ocr.ocr(pad_cell)
            ocr_result = sorted(ocr_result, key=lambda x: (x[0][0][0]))  # 排序 del_ocr_repeat_words前需要排序

            if len(ocr_result) > 0:
                num,_ = sort_paddle_result_by_pos(ocr_result)
                if place_name:  # 一行一行 识别
                    ocr_result = del_ocr_repeat_words(ocr_result)  # 元组转列表 空行
                    info = ' '.join([x[1][0] for x in ocr_result])  #
                    return info
        else:
            ocr_result = ocr.ocr(cell)  # 压缩造成字母识别不出来

        info = ''
        # 排序 获取行的数量 可能漏行
        if len(ocr_result) > 0:
            if row_num == 1 :# or place_name == 'yunnan':  # 只有一行 按照x排序 云南 批次比较高 get_y_arr_by_contours可能被识别成两行
                ocr_result = sorted(ocr_result, key=lambda x: (x[0][0][0]))
                # check_ocr_complete(cell, ocr_result)
                ocr_result = del_ocr_repeat_words(ocr_result)
                info = ' '.join([x[1][0] for x in ocr_result])

            else:  # 每一行单独OCR 因为OCR有时候识别不出中间部分的文字
                is_rec = 1
                ocr_rows_result = []#每一行单独OCR时保留的结果
                for i in range(len(y_subset_arr) - 1):
                    row_cell = cell[y_subset_arr[i]:y_subset_arr[i + 1], :]
                    row_ratio = row_cell.shape[1] // row_cell.shape[0]

                    if row_ratio >= 8:  #
                        padding_row_cell = np.pad(row_cell, ((10, 10), (0, 0)), 'constant', constant_values=(255))
                        ocr_row_result = ocr.ocr(padding_row_cell)  # 空格导致检测框问题
                    else:
                        ocr_row_result = ocr.ocr(row_cell)

                    if not ocr_row_result:#识别失败
                        is_rec = 0
                        row_rec_fail = i #识别失败的行数
                        break
                    else:
                        ocr_row_result = sorted(ocr_row_result, key=lambda x: (x[0][0][0])) #按照x排序 如果有多行可能出错
                        ocr_row_result = del_ocr_repeat_words(ocr_row_result) # 可能 '' 也会识别出来
                        # 用OCR结果判断是否真是一行
                        row_nums, ocr_row_result = sort_paddle_result_by_pos(ocr_row_result, row_max_span)  #row_height / 2
                        if row_nums > 1:
                            error_2rec1 =  ' '.join([x[1][0] for x in ocr_row_result])

                        ocr_rows_result.extend(ocr_row_result)

                 # 山西由于扫描质量问题 可能两行实际上是一行
                if is_rec:
                    info = ' '.join([x[1][0] for x in ocr_rows_result])
                else:
                    info = ' '.join([x[1][0] for x in ocr_result])
                    # logger.warn('{} 第{}行有多行数据，一行一行扫描时某一行无数据：{} 【一般是切割错误造成】'.format(file, i, info))
        return info


