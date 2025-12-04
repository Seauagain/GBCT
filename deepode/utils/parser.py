import argparse
import json

def parse_dict_arg(arg_string):
    """解析命令行传入的字典参数
    
    Args:
        arg_string (str): 字典字符串，如 '{"key1":"value1","key2":123}'
        
    Returns:
        dict: 解析后的字典
    """
    try:
        return json.loads(arg_string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("无效的JSON格式")

def parse_keys(config: dict, description: str = None):
    r"""Add the corresponding command line argument for each key in a Dict object.
    For config = {"num": 10}. Use `--num` in the command line.
    
    Args:
        config (dict): 配置字典，包含默认参数
        description (str, optional): 命令行参数描述
        
    Returns:
        argparse.Namespace: 解析后的参数对象
    # """

    parser = argparse.ArgumentParser(description=description)

    for key, value in config.items():
        if type(value) in [int, float, str]:
            parser.add_argument(f"--{key}", default=value, type=type(value))
        elif type(value) in [list, tuple]:
            if len(value) > 0:
                parser.add_argument(f"--{key}", nargs="+", default=value, type=type(value[0]))
            else:
                parser.add_argument(f"--{key}", nargs="+", default=value )
        elif type(value) in [bool]:
            # 布尔类型参数的处理改进
            parser.add_argument(f"--{key}", action="store_true" if not value else "store_false",
                               dest=key, default=value,)
        elif type(value) is dict:
            # 字典类型参数的处理
            parser.add_argument(f"--{key}", default=json.dumps(value), type=json.loads,
                               help=f"JSON格式的字典: {json.dumps(value)}")

        else:
            raise ValueError(f"got unexpected value type {type(value)} in the key='{key}'")
    
    args = parser.parse_args()
    
    # 将解析后的参数与原始配置合并
    for key, value in vars(args).items():
        config[key] = value
    
    return args


def dict_to_namespace(d: dict):
    """将字典转换为argparse.Namespace对象
    
    Args:
        d (dict): 输入字典
        
    Returns:
        argparse.Namespace: 转换后的Namespace对象
    """
    return argparse.Namespace(**d)


def namespace_to_dict(namespace: argparse.Namespace):
    """将argparse.Namespace对象转换为字典
    
    Args:
        namespace (argparse.Namespace): 输入的Namespace对象
        
    Returns:
        dict: 转换后的字典
    """
    return vars(namespace)


# 示例用法
if __name__ == "__main__":
    # 示例配置
    config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "model_name": "resnet50",
        "use_pretrained": True,
        "layers": [64, 128, 256, 512]
    }
    
    # 解析命令行参数
    args = parse_keys(config, description="模型训练参数")
    
    # 打印解析后的参数
    print("解析后的参数:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")