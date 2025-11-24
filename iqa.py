import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
from LOE import test

def comput(path,metric_name,dataset_name,method_name):
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input', type=str, default=r'E:\Project\MyProject\linux\Zero-Retinex\result\netrgb-twoStage-fusion-ab\loss\LIME-iter50-lr0.00100-lamba2.0.00001-gamma1.3\\', help='input image/folder path.')
    parser.add_argument('-i', '--input', type=str, default=path, help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=r'E:\dataset\LLIE\LOLdataset\val\high\\', help='reference image/folder path if needed.')
    parser.add_argument('--metric_mode', type=str, default='NR', help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('-m', '--metric_name', type=str, default=metric_name, help='IQA metric name, case sensitive.')
    parser.add_argument('--save_file', type=str, default=None, help='path to save results.')

    args = parser.parse_args()
    if dataset_name == 'LOL':
        args.ref = r'E:\dataset\LLIE\LOLdataset\val\high\\'
    elif dataset_name == 'LSRW':
        args.ref = r'E:\dataset\LLIE\LSRW-ori\Eval\gt\\'
    elif dataset_name == 'FiveK':
        args.ref = r'E:\dataset\LLIE\FiveK\test\target\\'
    elif dataset_name == 'SICE':
        args.ref = r'E:\dataset\LLIE\SICE\\'
    elif dataset_name == 'BAID':
        args.ref = r'E:\dataset\LLIE\BAID\test\output_resize\\'
    elif dataset_name == 'LOLv2':
        args.ref = r'E:\dataset\LLIE\LOLv2\Test\gt\\'
    elif dataset_name == 'UHD':
        args.ref = r'E:\dataset\LLIE\UHD\testing_set\gt\\'
    metric_name = args.metric_name.lower()
    if metric_name != 'loe':
        # set up IQA model
        iqa_model = create_metric(metric_name, metric_mode=args.metric_mode)
        metric_mode = iqa_model.metric_mode

        if os.path.isfile(args.input):
            input_paths = [args.input]
            if args.ref is not None:
                ref_paths = [args.ref]
        else:
            input_paths = sorted(glob.glob(os.path.join(args.input, '*')))
            if args.ref is not None:
                ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))

        if args.save_file:
            sf = open(args.save_file, 'w')

        avg_score = 0
        test_img_num = len(input_paths)
        if metric_name != 'fid':
            #pbar = tqdm(total=test_img_num, unit='image')
            for idx, img_path in enumerate(input_paths):
                #print(img_path)
                if 'Thumbs.db' in img_path:
                    continue
                img_name = os.path.basename(img_path)
                if metric_mode == 'FR':
                    ref_img_path = ref_paths[idx]
                else:
                    ref_img_path = None
                try:
                    score = iqa_model(img_path, ref_img_path).cpu().item()
                    avg_score += score
                    if args.save_file:
                        sf.write(f'{img_name}\t{score}\n')
                except AssertionError as e:
                    #os.remove(img_path)
                    print(e)

                # pbar.update(1)
                # pbar.set_description(f'{metric_name} of {img_name}: {score}')
                # pbar.write(f'{metric_name} of {img_name}: {score}')

            #pbar.close()
            avg_score /= test_img_num
        else:
            assert os.path.isdir(args.input), 'input path must be a folder for FID.'
            avg_score = iqa_model(args.input, args.ref)

        msg = f'Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}'
        #print(msg)
        print("%s,%s,%s"%(dataset_name,method_name,metric_name),":%.4f" % avg_score)

        if args.save_file:
            sf.write(msg + '\n')
            sf.close()

        if args.save_file:
            print(f'Done! Results are in {args.save_file}.')
        else:
            print(f'Done!')

    else:
        inp_path = r"E:\dataset\LLIE\testset\%s\*" % dataset_name
        avg_score = test(inp_path,path)
        print("%s,%s,%s"%(dataset_name,method_name,metric_name),":%.4f" % avg_score)

if __name__ == '__main__':
    metric_name = 'NIMA'
    path = r"E:\LLIE_expdata\data-name\DICM\ZRLLE-PQP"
    #dataset_list = ['MEF','DICM','NPE', 'exdark','VV','SICE',]
    #dataset_list = ['LIME','MEF','DICM','NPE', 'VV','LOL']
    dataset_list = ['LIME','DICM','NPE','Exdark']
    #dataset_list = ['LOL']'SNR',NPE
    #method_list = ['Zero-DCE++','SCI','Ours']
    #method_list = ['SNR','MBPNet','LANet','FourLLIE','CUE','K2FN']
    method_list = [
    'LIME',
'RRM',
'Zero-DCE++',
'SCI',
'URetinex',
'PairLIE',
'NeRCo',
'ZRLLE-PQP',
'Ours',
]

    #method_list = ['NeRCo']
    #method_list = ['LANet','CUE','NeRCo']
    #method_list = ['SNR','MBPNet','LANet','FourLLIE','CUE','Zero-DCE++','RUAS','SCI','PairLIE','NeRCo','ZRLLE-PQP']
    #method_list = ['MBPNet','LANet','FourLLIE','Zero-DCE++','CUE','PairLIE','NeRCo','ZRLLE-PQP','Ours']
    # method_name = "Zero-DCE++"
    # dataset_name = "NPE"
    for dataset_name in dataset_list:
        for method_name in method_list:
            enhenc_path = r"E:\LLIE_expdata\data-name\%s\%s" % (dataset_name, method_name)
            comput(enhenc_path,metric_name,dataset_name,method_name)
