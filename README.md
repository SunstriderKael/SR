# Super resolution toolbox

1.模型介绍
 ESRGAN/Meta-SR/SPSR, AdaFM, CARN/MCAN. 其中:
（1）重量级模型ESRGAN/Meta-SR/SPSR, 支持服务器端gpu部署;
（2）轻量级模型CARN/MCAN, 支持移动端模型移植.

2.训练结果
（1）ESRGAN使用mmsr工具箱进行训练, uban100数据集上的PSNR为24.6. 目前仅支持x4超分, 可修改mmsr工具箱的config文件进行不同尺度的训练;
（2）Meta-SR支持任意尺度的超分, 论文复现感知效果较差(不推荐);
（3）AdaFM适用于图像去模糊, 感知效果好;
（4）CARN支持x4/x3/x2, uban100数据集上的PSNR为30.42，感知效果好;
（5）MCAN支持x4/x3/x2, uban100数据集上的PSNR为30.03，感知效果好;

3.模型使用
  参见各个模块的readme, 基本都能调试通过.ESRGAN模块训练会用到mmsr工具箱, 需要熟悉mmsr.
