import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os

# original complete dataset: SCD30v1, SCD30v2
# separate SCD10 from SCD30v1
data_path = 'data'
if not os.path.exists(data_path):
    os.mkdir(data_path)
    
set_V2 = SPEECHCOMMANDS(root="data", download=True)                       # v2
set_V1 = SPEECHCOMMANDS(root="data", url='speech_commands_v0.01',download=True) # v1

# classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
# original_path = "data/speech_commands_v0.01"
# new_path = "data/SCD10"
# os.makedirs(new_path, exist_ok=True)

# # 遍历源数据集，将目标类别的样本复制到新数据集目录中
# for class_name in classes:
#     class_dir = os.path.join(original_path, class_name)
#     target_class_dir = os.path.join(new_path, class_name)
#     os.makedirs(target_class_dir, exist_ok=True)
    
#     # 将目标类别的前100个样本复制到新数据集目录中
#     for file_name in os.listdir(class_dir):
#         source_file = os.path.join(class_dir, file_name)
#         target_file = os.path.join(target_class_dir, file_name)
#         shutil.copyfile(source_file, target_file)
# print("SCD10 collection completed.")