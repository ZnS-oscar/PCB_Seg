class ComponentResult:
    """
    用于进行单个元件信息的存储
    单个元件对应的三册的数据结构就是元件的大类 - 元件的料号 - 元件的名称
    """
    def __init__(self, confidence, label_ID, center_x, center_y, width, height, class_name, part_name, component_name):
        """
        初始化单个元件信息
        :param confidence: 当前元件的识别的置信度
        :param label_ID: 当前元件的大类的类别 ID
        :param center_x: 当前识别的元件在原图的中心点 X
        :param center_y: 当前识别的元件在原图的中心点 Y
        :param width: 当前识别的元件的宽度
        :param height: 当前识别的元件的高度
        :param class_name: 当前识别的元件的大类  例如：电感、电容、电阻、引脚、焊盘等
        :param part_name: 当前识别的元件的料号名称  例如：电阻下面有：R0201、R01005 等
        :param component_name: 当前识别的元件的名称 例如 R0201 下面是板子上有 R0201 - 1 号电阻 R0201 - 2 号电阻
        """
        self.confidence = confidence
        self.label_ID = label_ID
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.class_name = class_name
        self.part_name = part_name
        self.component_name = component_name


class SegResult:
    """
    用于存放总的结果类
    所有的结果都存放在这里面，最后可以通过列表去遍历对应的结果即可
    """
    def __init__(self, mask, component_results=None):
        """
        初始化总的结果
        :param mask: cv mat 格式的推理时候放进去原图大小一致的 mask 图，对应的单个分割区域到元件名称级别，
                     主要用于进行检测时候的区域分割和显示，方便查看分割结果是否准确
        :param component_results: 存放每个元件的推理结果
        """
        self.mask = mask
        if component_results is None:
            self.component_results = []
        else:
            self.component_results = component_results