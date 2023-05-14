import os 
import zipfile
import requests
import numpy as np

def download_extract(url):
    """
    Download the  dataset.

    """
    # os.environ['KAGGLE_USERNAME'] = 'raj19084'
    # os.environ['KAGGLE_KEY'] = '511dac09104e9e7e3f51d8a32128e8a3'
    BASE_DIR = './data/';
    print(f"url={url}");

    os.makedirs(f'{BASE_DIR}', exist_ok=True)
    list_dirs = os.listdir("./")
    temp = [True if list_dirs[i].find("data")!=-1 else False for i in range(len(list_dirs))];
    temp = np.array(temp);
    base_dir = os.path.dirname(os.path.join(BASE_DIR,"anime"));
    if(temp.sum()>0):
        path_name = os.path.join(os.getcwd(),"data","Processed_Datasets","Anime","anime.zip");
        fp = zipfile.ZipFile(path_name, 'r');
        fp.extractall(base_dir)
        print('Extraction is Done')
        return;


    
    

    fname = os.path.join(BASE_DIR, url.split('/')[-1])
    
    print(f"fname={fname}");
    url = "https://github.com/caserec/Datasets-for-Recommender-Systems/archive/refs/heads/master.zip"
    os.makedirs(fname,exist_ok=True);
    main_path = os.getcwd();

    response = requests.get(url,stream=True,verify=False);

    if response.status_code == 200:
        
        print(f'Downloading {fname} from {url}...')
        # r = requests.get(url, stream=False, verify=False)
        # print(f"response_code={r.status_code} response={r.content}");
        with open(os.path.join(main_path,"data"), "wb") as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
        base_dir = os.path.dirname(fname)
        data_dir, ext = os.path.splitext(fname)

        if ext == '.zip':
            fp = zipfile.ZipFile(fname, 'r')
        fp.extractall(base_dir)
        print('Extract is Done')

