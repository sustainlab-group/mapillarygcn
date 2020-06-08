import shutil
import requests
import os
from tqdm import tqdm
import multiprocessing
import pandas as pd
import time

from PIL import Image


RESOLUTION = 2048
MAPILLARY_IM_URL = 'https://d1cuyjsrcm0gby.cloudfront.net/%s/thumb-%d.jpg'

def dl_images(entry):
    i, row = entry

    # get path for image, create dir if doesn't exist
    im_path = os.path.join('data/img_highres', '%s-%s.png' % (row['unique_cluster'], row['key']))

    if os.path.exists(im_path): # don't redownload
        return

    directory = os.path.dirname(im_path)
    try:
        os.makedirs(directory)
    except FileExistsError:
        # directory already exists
        pass

    # make request
    url = MAPILLARY_IM_URL % (row['key'], RESOLUTION)
    for attempt in range(7):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            break
        print(response.content, response.url)
        print('Retrying in 5 secs\n')
        time.sleep(5)
    else:
        print(response.content)
        exit()

    # write image
    with open(im_path, 'wb') as out_file:
        response.raw.decode_content = True #dont want compressed version
        shutil.copyfileobj(response.raw, out_file)
    del response

if __name__ == "__main__":
    img_df = pd.read_csv('data.csv')
    p = multiprocessing.Pool()
    for result in tqdm(p.imap_unordered(dl_images, img_df.iterrows(), chunksize=500), total=img_df.shape[0]):
        continue
    # once done downloading, add another column with image paths
    img_df_np = img_df.to_numpy()
    get_img_path = lambda row: 'data/img_highres/%s-%s.png' % (row[1], row[0])
    paths = np.apply_along_axis(get_img_path, 0, img_df_np)
    img_df['img_path'] = paths
    img_df.to_csv('data.csv', index=False)
