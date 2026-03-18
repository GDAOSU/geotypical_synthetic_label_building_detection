import os
import os.path as osp
from glob import glob
import shutil
from tqdm import tqdm

INPUTS = [r'E:\sythetic_city\CLAN_synrs\building_dataset\OSU_building',
       r'E:\sythetic_city\CLAN_synrs\building_dataset\OSMOSU2DR_building']

OUTPUT = r'E:\sythetic_city\CLAN_synrs\building_dataset\OSUSynOSUDR_building'
OUTPUT_IMG = osp.join(OUTPUT, 'imgs')
OUTPUT_MASK = osp.join(OUTPUT, 'masks')
os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_MASK, exist_ok=True)

for INPUT in tqdm(INPUTS):
    prefix = osp.basename(INPUT)+'_'
    INPUT_IMG = osp.join(INPUT, 'imgs')
    INPUT_MASK = osp.join(INPUT, 'masks')

    for i in os.listdir(INPUT_IMG):
        srcpath = osp.join(INPUT_IMG, i)
        dstpath = osp.join(OUTPUT_IMG, prefix+i)
        shutil.copy(srcpath,dstpath)

    for i in os.listdir(INPUT_MASK):
        srcpath = osp.join(INPUT_MASK, i)
        dstpath = osp.join(OUTPUT_MASK, prefix+i)
        shutil.copy(srcpath,dstpath)

    for txt in ['train.txt', 'val.txt', 'label.txt', 'all.txt']:
        open(osp.join(OUTPUT,txt),'a').writelines([prefix+i for i in open(osp.join(INPUT, txt), 'r')])
    shutil.copy(osp.join(INPUT,'info.json'), osp.join(OUTPUT,'info.json'))
