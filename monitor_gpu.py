import subprocess
import time
import re

output_file = 'gpu_usage.txt'
prev_gpu_info = None

while True:
    # 运行nvidia-smi命令并获取输出
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    gpu_info = result.stdout.decode('utf-8')
    gpu_info = gpu_info[gpu_info.index('\n', gpu_info.index('\n') + 1):]

    # 使用正则表达式找到日期行的位置并删除日期部分
    date_pattern = re.compile(r'\b[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}\b')
    match = date_pattern.search(gpu_info)
    if match:
        start_index = match.start()
        gpu_info = gpu_info[start_index + len(match.group()):]

    # 检查GPU使用情况是否发生变化
    if gpu_info != prev_gpu_info:
        # 将显卡使用情况写入txt文件
        with open(output_file, 'a') as file:
            file.write(gpu_info + '\n')
        
        prev_gpu_info = gpu_info

    time.sleep(1)  # 等待1秒后再次读取

