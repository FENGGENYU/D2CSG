import json
import glob
import os

import numpy as np

categories = ['chair', 'bed', 'table', 'storagefurniture', 'lamp']
combined_dict = {}
for cate in categories:
	json_file_path = os.path.join(f'./{cate}_exp', 'dev.json')
	with open(json_file_path, "r") as file:
		data = json.load(file)
		combined_dict.update(data)

out_file_name = os.path.join(f'./submission.json')
with open(out_file_name, 'w') as f:
	json.dump(combined_dict, f)
