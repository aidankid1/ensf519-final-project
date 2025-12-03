import torch

print("CUDA available:", torch.cuda.is_available())
num_gpus = torch.cuda.device_count()
print("GPU count:", num_gpus)

for i in range(num_gpus):
    print(f"\n===== GPU {i} =====")
    print("Name:", torch.cuda.get_device_name(i))
    print("Capability:", torch.cuda.get_device_capability(i))
    print("Total Memory (bytes):", torch.cuda.get_device_properties(i).total_memory)
    print("Multi-Processor Count:", torch.cuda.get_device_properties(i).multi_processor_count)
