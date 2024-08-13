import os
import sys
import cv2
from tqdm import tqdm
import sod_metrics as M

FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()


for _data_name in ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']:
    mask_root = 'C:/Users/Lenovo/Desktop/Datasets/TestDataset/{}/GT'.format(_data_name)
    pred_root = './results/{}/'.format(_data_name)
    mask_name_list = sorted(os.listdir(mask_root))


    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
    }

    print(results)
    file = open("res.txt", "a")
    file.write( ' ' + _data_name + ' ' + str(results) + '\n')




