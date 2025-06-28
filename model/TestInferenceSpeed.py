import torch
import time
import importlib
import os

# 模型文件名及标签
model_files = {
    "原始模型": "model_csp3",
    "替换卷积": "model_csp3_替换卷积",
    "移除渐进通道": "model_csp3_移除渐进式通道增量",
    "替换+移除": "model_csp3_替换和移除",
}

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备：{device}")

# 输入数据
input_tensor = torch.randn(1, 3, 640, 640).to(device)

# 每个模型测试次数
warmup = 5       # 预热次数
repeats = 20     # 计时次数

# 结果存储
timings = {}
    
for name, module_name in model_files.items():
    print(f"\n📦 正在加载模型: {module_name}.py")

    # 动态导入模块
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"❌ 找不到模型文件: {module_name}.py，请确认文件存在")
        continue

    # 初始化模型并移动到 GPU
    model = module.Net().to(device)
    model.eval()

    # 预热
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    # 正式计时
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / repeats * 1000  # 转换为 ms
    timings[name] = avg_time
    print(f"✅ 模型 {name} 推理平均耗时: {avg_time:.2f} ms")

# 排序输出
print("\n📊 所有模型推理速度对比（单位: ms）")
for name, t in sorted(timings.items(), key=lambda x: x[1]):
    print(f"{name:<10}: {t:.2f} ms")
