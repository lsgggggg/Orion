import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
import basicts

# 调试：打印 sys.path
print("sys.path:", sys.path)

torch.set_num_threads(4)

def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default="stid/PEMS04.py", help="training config")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 调试：尝试手动导入模块
    try:
        cfg_module = __import__(args.cfg.replace('/', '.')[:-3], fromlist=['CFG'])
        print("Imported CFG:", cfg_module.CFG)
    except Exception as e:
        print("Failed to import CFG:", str(e))
    
    # 启动训练，不依赖返回值
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)
    
    # 手动加载配置并执行测试
    cfg = cfg_module.CFG
    ckpt_save_dir = cfg['TRAIN']['CKPT_SAVE_DIR']
    model_name = cfg['MODEL']['NAME']
    best_model_path = os.path.join(
        ckpt_save_dir,
        f'{model_name}_best_val_MAE.pt'
    )
    
    # 调整配置以适配 easytorch 的键名要求
    cfg['MODEL.NAME'] = cfg['MODEL']['NAME']  # 添加扁平化键名
    
    # 创建 runner 实例并手动设置 ckpt_save_dir
    runner_class = cfg['RUNNER']
    runner = runner_class(cfg)
    runner.ckpt_save_dir = ckpt_save_dir  # 手动设置，避免依赖 get_ckpt_save_dir
    runner.logger.info('Training finished. Starting final test on the best model.')
    runner.load_model(ckpt_path=best_model_path, strict=True)
    runner.test_pipeline(cfg=cfg, train_epoch=None, save_metrics=True, save_results=True)