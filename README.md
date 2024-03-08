# 简介
extract_table 项目用于，识别图像中的表格，保存表格图像到指定文件夹，并提取表格的每一条横线，纵线坐标到数据库


## 近期更新
20240308 第一版本 妇女节礼物

下一期 更新目标：无表格框线的表格内容识别

## 特性
- 表格信息存入数据库，方便统计
  
- 文本识别通过轮廓获取每一行的开始和结束位置，一行一行识别提高了准确率，和排序正确性

- paddle ocr识别做了优化：
  - 通过padding,resize图像 等方法解决长文本识别为空问题
  - 通过比较坐标纠正了重复识别字符的问题
  - 通过切割单个数字所在图像+特殊参数设置提高单个数字识别率
  


## 快速开始
安装paddle paddle, paddle ocr ,opencv 3.4.1 
 
创建 gray_z_info 表保存表格信息

### 创建保存图像信息的表格 gray_z_info

创建mysql表：

CREATE TABLE `gray_z_info` (
  `省份` text,
  `原始路径` text,
  `原始文件夹` varchar(255) DEFAULT NULL,
  `保存表格图像路径` text,
  `图像高度` bigint(20) DEFAULT NULL,
  `图像宽度` bigint(20) DEFAULT NULL,
  `时间` text,
  `坐标` text,
  `列数` int(11) DEFAULT NULL,
  `行数` int(11) DEFAULT NULL,
  `行最大高度` int(11) DEFAULT NULL,
  `行最小高度` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

base_utils.py 包填入数据库账号密码
执行示例文件：
python test.py

如果你想识别自己的图像中的表格
修改main.py中的一系列常数
修改tools.params.py的ROW_HEIGHT 每一行文本高度的常数
 

# 👀 效果展示
原表格图像：

![img](https://github.com/dirac472/tableOCR/blob/main/image/1.jpg)

提取到表格图像：

![img](https://github.com/dirac472/tableOCR/blob/main/table_image/1_0.jpg) 

OCR识别结果：
见 result文件夹下excel

# 愿景
让你的工作或生活有点儿副产品
