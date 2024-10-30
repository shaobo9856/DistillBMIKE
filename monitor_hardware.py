import time
import psutil
import GPUtil
import os

def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return ["No GPU detected."]
        gpu_info = []
        for gpu in gpus:
            gpu_info.append(f"GPU {gpu.id}: {gpu.name}, Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB")
        return gpu_info
    except Exception as e:
        return [f"Error retrieving GPU info: {e}"]

def get_cpu_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_cores = psutil.cpu_count(logical=True)
    return f"CPU Usage: {cpu_percent}%, CPU Cores: {cpu_cores}"

def get_memory_info():
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 2)  # MB
    used_memory = memory.used / (1024 ** 2)  # MB
    free_memory = memory.available / (1024 ** 2)  # MB
    return f"Memory Total: {total_memory:.2f}MB, Used: {used_memory:.2f}MB, Free: {free_memory:.2f}MB"

def get_disk_info():
    disk_usage = psutil.disk_usage('/')
    total_disk = disk_usage.total / (1024 ** 3)  # GB
    used_disk = disk_usage.used / (1024 ** 3)  # GB
    free_disk = disk_usage.free / (1024 ** 3)  # GB
    return f"Disk Total: {total_disk:.2f}GB, Used: {used_disk:.2f}GB, Free: {free_disk:.2f}GB"

def print_hardware_info():
    while True:
        os.system('clear')  # 清屏（Linux/Mac），在 Windows 中可以使用 os.system('cls')
        print("Hardware Information:")
        print("\n".join(get_gpu_info()))
        print(get_cpu_info())
        print(get_memory_info())
        print(get_disk_info())
        print("="*50)
        time.sleep(5)

if __name__ == "__main__":
    print_hardware_info()
