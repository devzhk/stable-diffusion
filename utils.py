import os
import requests

url_dict = {
    'c256v2': 'https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt'
}

path_dict = {
    'c256v2': 'models/ldm/cin256-v2'
}


def download_file(url, file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 1024):
                f.write(chunk)
    print('Complete')


def download_model(name='c256v2'):
    url = url_dict[name]
    save_dir = path_dict[name]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.ckpt')
    download_file(url, save_path)