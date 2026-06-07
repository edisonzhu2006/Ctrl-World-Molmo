import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

def load_and_process_ann_file(data_root, ann_file, sequence_interval=1, start_interval=4, sequence_length=8):
    samples = []
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except:
        print(f'skip {ann_file}')
        return samples

    n_frames = ann['video_length']
    traj_len = int(sequence_length*sequence_interval)
    end_idx = n_frames - int(traj_len*0.5)
    if end_idx < 1:
        end_idx = 1

    for start_frame in range(0,end_idx,start_interval):       
        idx = start_frame
        sample = dict()
        sample['episode_id'] = ann['episode_id']
        sample['frame_ids'] = [idx]
        sample['states'] = np.array(ann['states'])[idx:idx+1]
        samples.append(sample)
    return samples

def init_anns(dataset_root, data_dir):
    final_path = f'{dataset_root}/{data_dir}'
    ann_files = [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith('.json')]
    return ann_files

def init_sequences(data_root, ann_files, sequence_interval, start_interval,sequence_length):
    samples = []
    with ThreadPoolExecutor(32) as executor:
        future_to_ann_file = {executor.submit(load_and_process_ann_file, data_root, ann_file, sequence_interval, start_interval, sequence_length): ann_file for ann_file in ann_files}
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_output_path', type=str, default='dataset_example/droid_subset')
    # dataset_name
    parser.add_argument('--dataset_name', type=str, default='droid_subset')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    ########################### xhand datasets ###########################
    sequence_length = 8
    # Process train before val so we can compute percentiles from train anchor states once.
    for data_type in ['train', 'val']:
        samples_all = []
        ann_files_all = []
        data_root = args.droid_output_path
        dataset_name = args.dataset_name

        sequence_interval = 1
        start_interval = 1
        ann_dir = f'annotation/{data_type}'
        ann_files = init_anns(data_root, ann_dir)
        ann_files_all.extend(ann_files)
        samples = init_sequences(data_root, ann_files,sequence_interval, start_interval, sequence_length)
        print(f'{data_root} {len(samples)} samples')
        samples_all.extend(samples)

        # 1% / 99% percentiles for Dataset_mix.normalize_bound (written once from TRAIN split).
        if data_type == 'train':
            print("########################### state ###########################")
            if len(samples_all) == 0:
                print("[warn] no train samples; skipping stat.json")
            else:
                state_all = []
                for s in samples_all:
                    state = np.asarray(s["states"], dtype=np.float64).squeeze(0)
                    state_all.append(state)
                state_all = np.stack(state_all, axis=0)
                print("state_all.shape", state_all.shape)
                state_01 = np.percentile(state_all, 1, axis=0)
                state_99 = np.percentile(state_all, 99, axis=0)
                print("state_01:", state_01)
                print("state_99:", state_99)
                stat = {
                    "state_01": state_01.tolist(),
                    "state_99": state_99.tolist(),
                }
                os.makedirs(f"dataset_meta_info/{dataset_name}", exist_ok=True)
                stat_path = f"dataset_meta_info/{dataset_name}/stat.json"
                with open(stat_path, "w") as f:
                    json.dump(stat, f)
                print(f"wrote {stat_path}")

        # dataset meta info
        for samples in samples_all:
            del samples['states']
        random.shuffle(samples_all)
        print('step_num',data_type,len(samples_all))
        print('traj_num',data_type, len(ann_files_all))
        os.makedirs(f'dataset_meta_info/{dataset_name}', exist_ok=True)
        with open(f'dataset_meta_info/{dataset_name}/{data_type}_sample.json', 'w') as f:
            json.dump(samples_all, f, indent=4)
        
