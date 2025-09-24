# 姿态特征文档

本文档详细描述了用于人体摔倒检测的姿态特征提取方法。

## 第一组特征：形状/分布特征

这些特征描述全身关键点的整体几何分布，用于捕捉人体姿态的宏观形状特性。

### 1. 扁平率 (flat_ratio)

**定义**: $$ flat_{ratio} = \frac{σy}{σx} $$

**计算方法**:
1. 计算关键点在竖直方向（y轴）的标准差 $ σy $
2. 计算关键点在水平方向（x轴）的标准差 $ σx $
3. 扁平率 = $ \frac{σy}{σx} $

**物理意义**:
- **站立姿态**: 人体在竖直方向上延展，y 向离散度大，$σy$ 较大 → $flat_{ratio}$较大
- **躺倒姿态**: 人体在水平方向上延展，x 向离散度大，$σy$ 较小 → $flat_{ratio}$较小

### 2. 跨度比 (y_span / x_span)

**定义**: 竖直跨度 ÷ 水平跨度

**计算方法**:
1. 计算所有关键点的y坐标最大值与最小值之差，得到 $y_{span}$
2. 计算所有关键点的x坐标最大值与最小值之差，得到 $x_{span}$
3. 跨度比 = $ \frac{y_{span}}{x_{span}} $

**物理意义**:
- **站立姿态**: 竖直跨度大，水平跨度小 → 比值大
- **躺倒姿态**: 水平跨度大，竖直跨度小 → 比值小

### 3. 重心与脚踝中心的距离

**定义**: 躯干中心与两脚踝中心点之间的欧氏距离

**计算方法**:
1. 计算躯干中心点：取左肩、右肩、左髋、右髋四个关键点的几何中心
   - $center_x = \frac{x_{left\_shoulder} + x_{right\_shoulder} + x_{left\_hip} + x_{right\_hip}}{4}$
   - $center_y = \frac{y_{left\_shoulder} + y_{right\_shoulder} + y_{left\_hip} + y_{right\_hip}}{4}$
2. 计算两脚踝的中心点
   - $ankle\_center_x = \frac{x_{left\_ankle} + x_{right\_ankle}}{2}$
   - $ankle\_center_y = \frac{y_{left\_ankle} + y_{right\_ankle}}{2}$
3. 计算躯干中心到脚踝中心的欧氏距离
   - $distance = \sqrt{(center_x - ankle\_center_x)^2 + (center_y - ankle\_center_y)^2}$

**物理意义**:
- **站立姿态**: 躯干中心在上方，脚踝中心在下方，距离较大
- **躺倒姿态**: 躯干中心与脚踝中心接近水平排列，距离可能增大（身体伸展）
- **坐姿**: 距离介于完全站立和躺倒之间

**分方向距离（可选）**:
- y方向距离: $d_y = |center_y - ankle\_center_y|$ - 反映垂直分离度
- x方向距离: $d_x = |center_x - ankle\_center_x|$ - 反映水平分离度

**组合使用**:
- 站立：y方向距离大，x方向距离小
- 躺倒：y方向距离小，x方向距离可能增大


### 4. 重心与头部的距离

**定义**: 躯干中心与头部中心在x和y方向上的距离

**计算方法**:
1. 计算躯干中心点（同特征3）：取左肩、右肩、左髋、右髋四个关键点的几何中心
   - $center_x = \frac{x_{left\_shoulder} + x_{right\_shoulder} + x_{left\_hip} + x_{right\_hip}}{4}$
   - $center_y = \frac{y_{left\_shoulder} + y_{right\_shoulder} + y_{left\_hip} + y_{right\_hip}}{4}$
2. 计算头部中心点：取鼻子、左眼、右眼、左耳、右耳五个关键点的几何中心
   - $head\_center_x = \frac{x_{nose} + x_{left\_eye} + x_{right\_eye} + x_{left\_ear} + x_{right\_ear}}{5}$
   - $head\_center_y = \frac{y_{nose} + y_{left\_eye} + y_{right\_eye} + y_{left\_ear} + y_{right\_ear}}{5}$
3. 计算分方向距离：
   - x方向距离: $d_x = |center_x - head\_center_x|$ - 反映头部水平偏移
   - y方向距离: $d_y = |center_y - head\_center_y|$ - 反映头部垂直偏移

**物理意义**:
- **站立姿态**: 头部在躯干正上方，y方向距离较大，x方向距离较小
- **躺倒姿态**: 头部与躯干接近水平排列，y方向距离较小，x方向距离可能增大
- **前倾/后仰**: x方向距离会明显变化，可识别身体倾斜方向
- **侧倒**: 根据倒向，x方向距离会有相应变化


### 5. 躯干与图像y轴的夹角

**定义**: 躯干主轴与图像垂直方向（y轴）之间的夹角

**计算方法**:
1. 确定躯干的两个端点：
   - 上端点：左肩和右肩的中点
     - $shoulder\_center_x = \frac{x_{left\_shoulder} + x_{right\_shoulder}}{2}$
     - $shoulder\_center_y = \frac{y_{left\_shoulder} + y_{right\_shoulder}}{2}$
   - 下端点：左髋和右髋的中点
     - $hip\_center_x = \frac{x_{left\_hip} + x_{right\_hip}}{2}$
     - $hip\_center_y = \frac{y_{left\_hip} + y_{right\_hip}}{2}$
2. 计算躯干向量：
   - $trunk\_vector = (shoulder\_center_x - hip\_center_x, shoulder\_center_y - hip\_center_y)$
3. 计算与y轴的夹角：
   - $\theta = arccos\left(\frac{|shoulder\_center_y - hip\_center_y|}{\sqrt{(shoulder\_center_x - hip\_center_x)^2 + (shoulder\_center_y - hip\_center_y)^2}}\right)$
   - 角度范围：0° ~ 90°

**物理意义**:
- **站立姿态**: 躯干垂直，夹角接近0°
- **躺倒姿态**: 躯干水平，夹角接近90°
- **倾斜姿态**: 夹角在0°~90°之间，值越大表示倾斜越严重
- **摔倒过程**: 夹角从小变大是摔倒的重要特征


### 6. 左右大腿与图像y轴的夹角

**定义**: 左右大腿分别与图像垂直方向（y轴）之间的夹角

**计算方法**:
1. 左大腿夹角：
   - 大腿向量：从左髋指向左膝
   - $left\_thigh\_vector = (x_{left\_knee} - x_{left\_hip}, y_{left\_knee} - y_{left\_hip})$
   - 与y轴夹角：
   - $\theta_{left\_thigh} = arccos\left(\frac{|y_{left\_knee} - y_{left\_hip}|}{\sqrt{(x_{left\_knee} - x_{left\_hip})^2 + (y_{left\_knee} - y_{left\_hip})^2}}\right)$

2. 右大腿夹角：
   - 大腿向量：从右髋指向右膝
   - $right\_thigh\_vector = (x_{right\_knee} - x_{right\_hip}, y_{right\_knee} - y_{right\_hip})$
   - 与y轴夹角：
   - $\theta_{right\_thigh} = arccos\left(\frac{|y_{right\_knee} - y_{right\_hip}|}{\sqrt{(x_{right\_knee} - x_{right\_hip})^2 + (y_{right\_knee} - y_{right\_hip})^2}}\right)$

**物理意义**:
- **站立姿态**: 大腿垂直，夹角接近0°
- **坐姿**: 大腿水平，夹角接近90°
- **躺倒姿态**: 大腿可能呈各种角度，取决于躺倒姿势
- **行走/跑步**: 左右大腿夹角会有明显差异，表示步态


### 7. 左右小腿与图像y轴的夹角

**定义**: 左右小腿分别与图像垂直方向（y轴）之间的夹角

**计算方法**:
1. 左小腿夹角：
   - 小腿向量：从左膝指向左脚踝
   - $left\_shin\_vector = (x_{left\_ankle} - x_{left\_knee}, y_{left\_ankle} - y_{left\_knee})$
   - 与y轴夹角：
   - $\theta_{left\_shin} = arccos\left(\frac{|y_{left\_ankle} - y_{left\_knee}|}{\sqrt{(x_{left\_ankle} - x_{left\_knee})^2 + (y_{left\_ankle} - y_{left\_knee})^2}}\right)$

2. 右小腿夹角：
   - 小腿向量：从右膝指向右脚踝
   - $right\_shin\_vector = (x_{right\_ankle} - x_{right\_knee}, y_{right\_ankle} - y_{right\_knee})$
   - 与y轴夹角：
   - $\theta_{right\_shin} = arccos\left(\frac{|y_{right\_ankle} - y_{right\_knee}|}{\sqrt{(x_{right\_ankle} - x_{right\_knee})^2 + (y_{right\_ankle} - y_{right\_knee})^2}}\right)$

**物理意义**:
- **站立姿态**: 小腿垂直，夹角接近0°
- **坐姿**: 小腿通常垂直或稍微倾斜，夹角0°~30°
- **躺倒姿态**: 小腿水平，夹角接近90°
- **屈膝状态**: 夹角变化反映膝盖弯曲程度


