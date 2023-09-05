# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from typing import Callable, List, Dict, Optional
from multiprocessing import Pool, Manager, Value, Lock

import glb_utils
import decimate
from tqdm import tqdm

output_dir = "data/hitl_simplified/data/"
omit_black_list = False
omit_gray_list = False

def file_is_scene_config(filepath: str) -> bool:
    """
    Return whether or not the file is an scene_instance.json
    """
    return filepath.endswith(".scene_instance.json")

def file_is_glb(filepath:str)->bool:
    """
    Return whether or not the file is a glb.
    """
    return filepath.endswith(".glb")

def find_files(root_dir: str, discriminator: Callable[[str], bool]) -> List[str]:
    """
    Recursively find all filepaths under a root directory satisfying a particular constraint as defined by a discriminator function.

    :param root_dir: The root directory for the recursive search.
    :param discriminator: The discriminator function which takes a filepath and returns a bool.

    :return: The list of all absolute filepaths found satisfying the discriminator.
    """
    filepaths: List[str] = []

    if not os.path.exists(root_dir):
        print(" Directory does not exist: " + str(dir))
        return filepaths

    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            sub_dir_filepaths = find_files(entry_path, discriminator)
            filepaths.extend(sub_dir_filepaths)
        # apply a user-provided discriminator function to cull filepaths
        elif discriminator(entry_path):
            filepaths.append(entry_path)
    return filepaths

def get_model_ids_from_scene_instance_json(filepath: str) -> List[str]:
    """
    Scrape a list of all unique model ids from the scene instance file.
    """
    assert filepath.endswith(".scene_instance.json"), "Must be a scene instance JSON."

    model_ids = []

    with open(filepath, "r") as f:
        scene_conf = json.load(f)
        if "object_instances" in scene_conf:
            for obj_inst in scene_conf["object_instances"]:
                model_ids.append(obj_inst["template_name"])
        else:
            print("No object instances field detected, are you sure this is scene instance file?")

    print(f" {filepath} has {len(model_ids)} object instances.")
    model_ids = list(set(model_ids))
    print(f" {filepath} has {len(model_ids)} unique objects.")

    return model_ids


def process_model(args):
    model_path, counter, lock, total_models = args
    
    if "/fphab/" in model_path and not "/stages/" in model_path:
        parts = model_path.split('/objects/')
        assert len(parts) == 2
        object_partial_filepath = parts[1]

        if "decomposed" in model_path:
            source_file = "/home/mdcote/Documents/git/fpss/objects/" + object_partial_filepath
        else:
            source_file = "/home/mdcote/Documents/git/fp-models/" + object_partial_filepath
        dest_file = output_dir + "fphab/objects/" + object_partial_filepath
    else:
        # works for ycb and hab_spot_arm meshes, and probably HSSD stage
        source_file = model_path
        dest_file = output_dir + model_path[5:]
        object_partial_filepath = model_path

    dest_directory = os.path.dirname(dest_file)

    if os.path.isfile(dest_file):
        print(f"Skipping:   {source_file}")
        result = {}
        result['status'] = "skipped"
        return result
    

    print(f"Processing: {source_file}")
    #print(f"dest_file: {dest_file}")

    # Create all necessary subdirectories
    os.makedirs(dest_directory, exist_ok=True)

    try:
        source_tris, target_tris, simplified_tris = \
            decimate.decimate(source_file, dest_file, quiet=True, sloppy=False)
    except:
        try:
            print(f"Unable to decimate: {source_file}. Trying sloppy.")
            source_tris, target_tris, simplified_tris = \
                decimate.decimate(source_file, dest_file, quiet=True, sloppy=True)
        except:
            print(f"Unable to decimate: {source_file}")
            result = {}
            result['status'] = "error"
            return result

    print(f"object_partial_filepath: {object_partial_filepath}")
    print(f"source_tris: {source_tris}, target_tris: {target_tris}, simplified_tris: {simplified_tris}")

    result = {
        'source_tris': source_tris,
        'simplified_tris': simplified_tris,
        'object_partial_filepath': object_partial_filepath,
        'status': "ok"
    }

    if simplified_tris > target_tris * 2 and simplified_tris > 3000:
        result['list_type'] = 'black'
        if omit_black_list:
            os.remove(dest_file)
    elif simplified_tris > 4000:
        result['list_type'] = 'gray'
        if omit_gray_list:
            os.remove(dest_file)
    else:
        result['list_type'] = None

    with lock:
        counter.value += 1
        print(f"{counter.value} out of {total_models} models have been processed so far")

    return result


def simplify_models(model_filepaths):

    total_source_tris = 0
    total_simplified_tris = 0
    black_list = []
    gray_list = []
    black_list_tris = 0
    gray_list_tris = 0
    total_skipped = 0
    total_error = 0

    # Initialize counter and lock
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    total_models = len(model_filepaths)

    # Pair up the model paths with the counter and lock
    args_lists = [(model_path, counter, lock, total_models) for model_path in model_filepaths]

    results = []

    use_multiprocessing = False  # total_models > 6
    if use_multiprocessing:
        max_processes = 6
        with Pool(processes=min(max_processes, total_models)) as pool:
            results = list(pool.map(process_model, args_lists))
    else:
        for args in args_lists:
            results.append(process_model(args))

    for result in results:
        if result['status'] == "ok":
            total_source_tris += result['source_tris']
            total_simplified_tris += result['simplified_tris']
            if result['list_type'] == 'black':
                black_list.append(result['object_partial_filepath'])
                black_list_tris += result['simplified_tris']
            elif result['list_type'] == 'gray':
                gray_list.append(result['object_partial_filepath'])
                gray_list_tris += result['simplified_tris']
        elif result['status'] == "error":
            total_error += 1
        elif result['status'] == "skipped":
            total_skipped += 1

    if (total_skipped > 0):
        print(f"Skipped {total_skipped} files.")
    if (total_error > 0):
        print(f"Skipped {total_error} files due to processing errors.")
    print(f"Reduced total vertex count from {total_source_tris} to {total_simplified_tris}")
    print(f"Without black list: {total_simplified_tris - black_list_tris}")
    print(f"Without gray and black list: {total_simplified_tris - black_list_tris - gray_list_tris}")

    for (i, curr_list) in enumerate([black_list, gray_list]):
        print("")
        print("black list" if i == 0 else "gray list" + " = [")
        for item in curr_list:
            print("    " + item + ",")
            # print("https://huggingface.co/datasets/fpss/fphab/blob/main/objects/" + item)
        print("]")


def add_models_from_scenes(dataset_root_dir, scene_ids, model_filepaths):

    fp_root_dir = dataset_root_dir
    config_root_dir = os.path.join(fp_root_dir, "scenes-uncluttered")
    configs = find_files(config_root_dir, file_is_scene_config)
    obj_root_dir = os.path.join(fp_root_dir, "objects")
    glb_files = find_files(obj_root_dir, file_is_glb)
    render_glbs = [f for f in glb_files if (".collider" not in f and ".filteredSupportSurface" not in f)]

    for filepath in configs:
        #these should be removed, but screen them for now
        if "orig" in filepath:
            print(f"Skipping alleged 'original' instance file {filepath}")
            continue
        for scene_id in scene_ids:
            #NOTE: add the extension back here to avoid partial matches
            if scene_id+".scene_instance.json" in filepath:
                print(f"filepath '{filepath}' matches scene_id '{scene_id}'")
                model_ids = get_model_ids_from_scene_instance_json(filepath)
                for model_id in model_ids:
                    for render_glb in render_glbs:
                        if model_id+".glb" in render_glb:
                            if "part" in render_glb and "part" not in model_id:
                                continue
                            # perf todo: consider a set instead of a list
                            if render_glb not in model_filepaths:
                                model_filepaths.append(render_glb)


def main():
    parser = argparse.ArgumentParser(
        description="Get all .glb render asset files associated with a given scene."
    )
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        help="path to HSSD SceneDataset root directory containing 'fphab-uncluttered.scene_dataset_config.json'.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        help="one or more scene ids",
    )

    model_filepaths = []

    args = parser.parse_args()
    scene_ids = list(dict.fromkeys(args.scenes)) if args.scenes else []
    add_models_from_scenes(args.dataset_root_dir, scene_ids, model_filepaths)

    # add stage
    for scene_id in scene_ids:
        model_filepaths.append(os.path.join(args.dataset_root_dir, "stages", scene_id + ".glb"))
        

    # todo: get these from episode set
    model_filepaths.append("data/objects/ycb/meshes/003_cracker_box/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/005_tomato_soup_can/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/024_bowl/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/025_mug/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/009_gelatin_box/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/010_potted_meat_can/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/007_tuna_fish_can/google_16k/textured.glb")
    model_filepaths.append("data/objects/ycb/meshes/002_master_chef_can/google_16k/textured.glb")


    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/base.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/fl.hip.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/fl.uleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/fl.lleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/fr.hip.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/fr.uleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/fr.lleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/hl.hip.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/hl.uleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/hl.lleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/hr.hip.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/hr.uleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/hr.lleg.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_sh0.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_sh1.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_hr0.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_el0.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_el1.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_wr0.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_wr1.glb")
    model_filepaths.append("data/robots/hab_spot_arm/urdf/../meshesColored/arm0.link_fngr.glb")


    simplify_models(model_filepaths)

if __name__ == "__main__":
    main()