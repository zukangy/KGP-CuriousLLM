import os 


if __name__ == '__main__':

    root_dir = '.'

    os.makedirs(os.path.join(root_dir, 'DATA/KG'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'DATA/2WikiMQA'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'DATA/HotpotQA'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'models'), exist_ok=True)