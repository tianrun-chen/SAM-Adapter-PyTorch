import gc
import sys
import torch

def show_memory():
        total = 0
        count = 0
        diff_size = set()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    total += sys.getsizeof(obj.storage())/1e6
                    count += 1
                    diff_size.add(sys.getsizeof(obj.storage())/1e6)
            except:
                pass
        
        print(f'summe: {total}')
        print(f'of {count} tensors with {len(diff_size)} different sizes')
        print(f'unique sizes: {sum(diff_size)}')