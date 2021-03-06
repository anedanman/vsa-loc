from pathlib import Path
import h5py
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import vsa
from sklearn.metrics.pairwise import cosine_similarity
import ast 
from sceneproc import create_im, sem2vec, sem2edgevec, sem2depthvec
import json
import matplotlib.pyplot as plt
import ray


def quaternion_to_rotation_matrix(quaternion_wxyz):
    r = R.from_quat([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    matrix = r.as_matrix()
    matrix[:3,2] = -matrix[:3,2]
    matrix[:3,1] = -matrix[:3,1]
    return matrix


def get_point_3d(x, y, depth, fx, fy, cx, cy, cam_center_world, R_world_to_cam, w_in_quat_first = True):
    if depth <= 0:
        return 0
    new_x = (x - cx)*depth/fx
    new_y = (y - cy)*depth/fy
    new_z = depth
    coord_3D_world_to_cam = np.array([new_x, new_y, new_z], float)
    if len(R_world_to_cam) == 4:
        if w_in_quat_first:
            matrix = quaternion_to_rotation_matrix(R_world_to_cam)
        else:
            R_world_to_cam = [R_world_to_cam[3], R_world_to_cam[0], R_world_to_cam[1], R_world_to_cam[2]]
            matrix = quaternion_to_rotation_matrix(R_world_to_cam)
    coord_3D_cam_to_world = np.matmul(matrix, coord_3D_world_to_cam) + cam_center_world
    return coord_3D_cam_to_world


def is_database(name):
    is_database = ''
    if name.find('database') != -1:
        is_database = '_base'
    return is_database


def proccess_descr(path_to_global_descriptors, query_image, database, sem_path, k=0.5, mode=''):
    sims = []
    with open(os.path.join(path_to_global_descriptors, query_image+'.npy'), 'rb') as f:
            descriptor_query = np.load(f)
    with open(os.path.join(sem_path, query_image+'.npy'), 'rb') as f:
            query_vec = np.load(f)        
    
    for db_image in database:
        with open(os.path.join(path_to_global_descriptors, db_image+'.npy'), 'rb') as f:
            descriptor2 = np.load(f) 
        with open(os.path.join(sem_path, db_image+'.npy'), 'rb') as f:
            db_vec = np.load(f)

        sim1 = cosine_similarity(descriptor_query.reshape(1, -1), descriptor2.reshape(1, -1))[0][0]
        sim2 = vsa.sim(query_vec, db_vec)
        sim = k * sim1 + (1-k) * sim2
        if mode == 'semantic':
            sim = sim2
        if mode == 'default':
            sim = sim1
        sims.append(sim)
        
    num_new = np.argmax(sims)
    db_image = database[num_new]
    simil_score = sims[num_new]
    return simil_score, db_image


def proccess_descr_adv(path_to_hdf_files, path_to_global_descriptors,
                       query_image, database, sem_path, k=0.5, n=5, hfnthr=0.1, semthr=0.1, res_name=''):
    with open(os.path.join(path_to_global_descriptors, query_image+'.npy'), 'rb') as f:
            descriptor_query = np.load(f)
    with open(os.path.join(sem_path, query_image+'.npy'), 'rb') as f:
            query_vec = np.load(f)
    hfnet_feat = np.zeros((len(database), descriptor_query.shape[0])) 
    sem_feat = np.zeros((len(database), query_vec.shape[0]))   
    dists = []
    db_names = []
    for i, db_image in enumerate(database):
        with open(os.path.join(path_to_global_descriptors, db_image+'.npy'), 'rb') as f:
            descriptor2 = np.load(f) 
        with open(os.path.join(sem_path, db_image+'.npy'), 'rb') as f:
            db_vec = np.load(f)
        hfnet_feat[i, :] = descriptor2
        sem_feat[i, :] = db_vec
        dist = get_dist(query_image, db_image, path_to_hdf_files)
        db_names.append(db_image)
        dists.append(dist)

    hfnetsims = (hfnet_feat @ descriptor_query.reshape(-1, 1)).reshape(-1)
    semsims = (sem_feat @ query_vec.reshape(-1, 1)).reshape(-1)
    result = [
        {'db_filename': db_image,
        'hfnetsim': sim1,
        'semsim': sim2,
        'dist': dist} for db_image, sim1, sim2, dist in zip(db_names, hfnetsims, semsims, dists)]

    if not os.path.exists(os.path.join(path_to_global_descriptors, 'res_' + res_name)):
        os.mkdir(os.path.join(path_to_global_descriptors, 'res_' + res_name))
    with open(os.path.join(path_to_global_descriptors, 'res_' + res_name, query_image+'.json'), 'w') as f:
        json.dump(result, f)

    tophfnet = sorted(result, key=lambda x: -x['hfnetsim'])
    topsem = []
    topdist = sorted(result, key=lambda x: x['dist'])
    topsem = sorted(result, key=lambda x: -x['semsim'])   

    res_ans = tophfnet[0]
    for pos_ans in tophfnet[1:n]:
        if res_ans['hfnetsim'] - pos_ans['hfnetsim'] < hfnthr and pos_ans['semsim'] - res_ans['semsim'] > semthr:
            res_ans = pos_ans
    db_image = res_ans['db_filename']
    delta = res_ans['dist'] - topdist[0]['dist']
    return delta, db_image, tophfnet[:n], topdist[:n], topsem[:n]


def adv_test(path_to_hdf='./datasets/Habitat/HPointLoc/',
              path_to_hfnet='./vectors/HF-Net_descriptors',
              path_to_sem='./vectors/semantic_edges',
              k=0.9, dist_thr=20, hfnthr=0.1, semthr=0.1, n=5, use_saved=True, res_name=''):
    results = []
    deltas = []
    problem_cases = []
    for map_name in os.listdir(path_to_hfnet):
        #creating database
        queries = []
        database = []
        for name in os.listdir(os.path.join(path_to_hfnet, map_name)):
            if 'res' in name:
                continue
            if is_database(name):
                database.append(name.rstrip('.npy'))
            else:
                queries.append(name.rstrip('.npy'))
                
        distances = [0.25, 0.5, 1, 5, 10, 20]
        res = {i:0 for i in distances}
        for query in tqdm(queries):
            if not use_saved:
                delta, db_image, tophfnet, topdist, topsem = proccess_descr_adv(os.path.join(path_to_hdf, map_name),
                                                                                os.path.join(path_to_hfnet, map_name),
                                                                                query, database,
                                                                                os.path.join(path_to_sem, map_name),
                                                                                k=k, n=n, semthr=semthr, hfnthr=hfnthr, res_name=res_name)
                dist_diff = get_dist(query, db_image, os.path.join(path_to_hdf, map_name))
                deltas.append(delta)
            else:
                with open(os.path.join(os.path.join(path_to_hfnet, map_name), 'res_' + res_name, query + '.json'), 'r') as f:
                    result = json.load(f)
                tophfnet = sorted(result, key=lambda x: -x['hfnetsim'])
                topdist = sorted(result, key=lambda x: x['dist'])
                topsem = sorted(result, key=lambda x: -x['semsim'])   
                res_ans = tophfnet[0]
                for pos_ans in tophfnet[1:n]:
                    st1 = (res_ans['hfnetsim'] - pos_ans['hfnetsim']) * 17 < (pos_ans['semsim'] - res_ans['semsim'])
                    st2 = pos_ans['hfnetsim'] >= 0.8 and (res_ans['hfnetsim'] - pos_ans['hfnetsim']) * 9 < (pos_ans['semsim'] - res_ans['semsim'])
                    if (res_ans['hfnetsim'] - pos_ans['hfnetsim'] <= hfnthr and pos_ans['semsim'] - res_ans['semsim'] >= semthr) or st1 or st2:
                        res_ans = pos_ans
                db_image = res_ans['db_filename']
                delta = res_ans['dist'] - topdist[0]['dist']
                dist_diff = get_dist(query, db_image, os.path.join(path_to_hdf, map_name))
                deltas.append(delta)

            for d in distances:
                if dist_diff < d:
                    res[d] += 1
            if dist_diff > dist_thr:
                problem_cases.append({'query_name': query, 'top_hfnet': tophfnet,
                                    'top_dist': topdist, 'top_sem': topsem, 'resans': res_ans})
        for i in res:
            res[i] = res[i] / len(queries)
        results.append(res)
        
    answer = {i:0 for i in distances}
    for res in results:
        for i in res:
            answer[i] += res[i]
    for i in answer:
        answer[i] = answer[i] / len(results)
    return answer, problem_cases, np.mean(deltas)
    
    
def get_pose(filename, path_to_hdf5_datasets):
    query1_filename = filename.rstrip('.npy')
    hdf5_filename_query1 = '_'.join(query1_filename.split('_')[:2]) + '.hdf5'
    hdf5_file_query1 = h5py.File(os.path.join(path_to_hdf5_datasets, hdf5_filename_query1), 'r')
    num_query1 = int(query1_filename.split('_')[-1])
    
    is_db = is_database(query1_filename)
    translation = hdf5_file_query1['gps'+is_db][num_query1]
    orientation = quaternion_to_rotation_matrix(hdf5_file_query1['quat'+is_db][num_query1])
    pose_44_query1 = np.eye(4)
    pose_44_query1[:3,:3] = orientation
    pose_44_query1[:3,3] = translation
    semantic = hdf5_file_query1['semantic'+is_db][num_query1]
    depth = hdf5_file_query1['depth'+is_db][num_query1]
    
    index_to_title_map_query = ast.literal_eval(str(np.array(hdf5_file_query1['index_to_title_map'])))
    mapping_query = np.array(hdf5_file_query1['mapping'])
        
    return depth, semantic, index_to_title_map_query, mapping_query, pose_44_query1


def poses_diff(pose1, pose2):
    diff_pose = np.linalg.inv(pose2) @ pose1
    dist_diff = np.sum(diff_pose[:3, 3]**2) ** 0.5
    r = R.from_matrix(diff_pose[:3, :3])
    rotvec = r.as_rotvec()
    angle_diff = (np.sum(rotvec**2)**0.5) * 180 / 3.14159265353
    angle_diff = abs(180 - abs(angle_diff-180))
    return dist_diff, angle_diff


def save_vectrs(descrs_path, sem_path, path_to_hdf5_datasets):
    im = None
    for name in tqdm(os.listdir(descrs_path)):
        _, db_sem, index_to_title_map, mapping, _ = get_pose(name, path_to_hdf5_datasets)
        if im is None:
            im = create_im(index_to_title_map.values())
            
        db_vec = sem2vec(db_sem, mapping, index_to_title_map, im)
        with open(os.path.join(sem_path, name), 'wb') as f:
            np.save(f, db_vec)
            
            
def save_edge_vectrs(descrs_path, sem_path, path_to_hdf5_datasets, use_R=True, skip_wall=True):
    im = None
    for name in tqdm(os.listdir(descrs_path)):
        _, db_sem, index_to_title_map, mapping, _ = get_pose(name, path_to_hdf5_datasets)
        if im is None:
            im = create_im(index_to_title_map.values())
            
        db_vec = sem2edgevec(db_sem, mapping, index_to_title_map, im, use_R, skip_wall)
        with open(os.path.join(sem_path, name), 'wb') as f:
            np.save(f, db_vec)


def save_depth_vectrs(descrs_path, sem_path, path_to_hdf5_datasets):
    im = None
    for name in tqdm(os.listdir(descrs_path)):
        depth_im, db_sem, index_to_title_map, mapping, _ = get_pose(name, path_to_hdf5_datasets)
        if im is None:
            im = create_im(index_to_title_map.values())
            
        db_vec = sem2depthvec(db_sem, depth_im, mapping, index_to_title_map, im)
        if not os.path.exists(sem_path):
            os.mkdir(sem_path)
        with open(os.path.join(sem_path, name), 'wb') as f:
            np.save(f, db_vec)


@ray.remote
def save_one_depth(sem_path, path_to_hdf5_datasets, name, im):
    depth_im, db_sem, index_to_title_map, mapping, _ = get_pose(name, path_to_hdf5_datasets)
    db_vec = sem2depthvec(db_sem, depth_im, mapping, index_to_title_map, im)
    if not os.path.exists(sem_path):
        os.mkdir(sem_path)
    with open(os.path.join(sem_path, name), 'wb') as f:
        np.save(f, db_vec)


def save_depth_vectrs_fast(descrs_path, sem_path, path_to_hdf5_datasets):
    name0 = os.listdir(descrs_path)[0]
    depth_im, db_sem, index_to_title_map, mapping, _ = get_pose(name0, path_to_hdf5_datasets)
    im = create_im(index_to_title_map.values())
    futures = [save_one_depth.remote(sem_path, path_to_hdf5_datasets, name, im) for name in os.listdir(descrs_path) if not 'res' in name]
    ray.get(futures)
            
            
def gl_save_vectrs(descrs_path='./vectors/HF-Net_descriptors',
                   sem_path='./vectors/semantic',
                   path_to_hdf5_datasets='./datasets/Habitat/HPointLoc/'):
    for map_name in os.listdir(descrs_path):
        save_vectrs(os.path.join(descrs_path, map_name),
                    os.path.join(sem_path, map_name),
                    os.path.join(path_to_hdf5_datasets, map_name))
        
        
def gl_save_edge_vectrs(descrs_path='./vectors/HF-Net_descriptors',
                        sem_path='./vectors/semantic_edges',
                        path_to_hdf5_datasets='./datasets/Habitat/HPointLoc/',
                        use_R=True, skip_wall=False):
    for map_name in os.listdir(descrs_path):
        save_edge_vectrs(os.path.join(descrs_path, map_name), 
                    os.path.join(sem_path, map_name),
                    os.path.join(path_to_hdf5_datasets, map_name),
                    use_R=use_R, skip_wall=skip_wall)


def gl_save_depth_vectrs(descrs_path='./vectors/HF-Net_descriptors',
                        sem_path='./vectors/semantic_depths',
                        path_to_hdf5_datasets='./datasets/Habitat/HPointLoc/'):
    if not os.path.exists(sem_path):
        os.mkdir(sem_path)
    for map_name in os.listdir(descrs_path):
        save_depth_vectrs(os.path.join(descrs_path, map_name), 
                    os.path.join(sem_path, map_name),
                    os.path.join(path_to_hdf5_datasets, map_name))


def gl_save_depth_vectrs_fast(descrs_path='./vectors/HF-Net_descriptors',
                        sem_path='./vectors/semantic_depths',
                        path_to_hdf5_datasets='./datasets/Habitat/HPointLoc/'):
    ray.init()
    if not os.path.exists(sem_path):
        os.mkdir(sem_path)
    for map_name in tqdm(os.listdir(descrs_path)):
        save_depth_vectrs_fast(os.path.join(descrs_path, map_name), 
                    os.path.join(sem_path, map_name),
                    os.path.join(path_to_hdf5_datasets, map_name))


def main_test(path_to_hdf='./datasets/Habitat/HPointLoc/',
              path_to_hfnet='./vectors/HF-Net_descriptors',
              path_to_sem='./vectors/semantic',
              mode='', k=0.9):
    results = []
    for map_name in os.listdir(path_to_hfnet):
        #creating database
        queries = []
        database = []
        for name in os.listdir(os.path.join(path_to_hfnet, map_name)):
            if is_database(name):
                database.append(name.rstrip('.npy'))
            else:
                queries.append(name.rstrip('.npy'))
                
        distances = [0.25, 0.5, 1, 5, 10, 20]
        res = {i:0 for i in distances}
        for query in tqdm(queries):
            simil_score, db_image = proccess_descr(os.path.join(path_to_hfnet, map_name),
                                                   query, database,
                                                   os.path.join(path_to_sem, map_name),
                                                   mode=mode, k=k)
            dist_diff = get_dist(query, db_image, os.path.join(path_to_hdf, map_name))
            for d in distances:
                if dist_diff < d:
                    res[d] += 1
        for i in res:
            res[i] = res[i] / len(queries)
        results.append(res)
        
    answer = {i:0 for i in distances}
    for res in results:
        for i in res:
            answer[i] += res[i]
    for i in answer:
        answer[i] = answer[i] / len(results)
    return answer
    

def get_dist(query_img, database_img, path_to_hdf_files):
    _, _, _, _, query_pose = get_pose(query_img, path_to_hdf_files)
    _, _, _, _, db_pose = get_pose(database_img, path_to_hdf_files)
    dist_diff, _ = poses_diff(query_pose, db_pose)
    return dist_diff


def getrgb(name, path_to_hdf='./datasets/Habitat/HPointLoc/'):
    maps = os.listdir(path_to_hdf)
    map_name = name.split('_')[0]
    for cur_map in maps:
        if map_name in cur_map:
            map_name = cur_map
    hdf5_filename = '_'.join(name.split('_')[:2]) + '.hdf5'
    hdf5_file = h5py.File(os.path.join(os.path.join(path_to_hdf, map_name), hdf5_filename), 'r')
    num = int(name.split('_')[-1])
    is_db = is_database(name)
    rgb = hdf5_file['rgb'+is_db][num]
    return rgb


def visual(case_log):
    def a(num):
        return int(num*100) / 100
    
    def proc_nums(d):
        return f"hfnet: {a(d['hfnetsim'])}, sem: {a(d['semsim'])}, dist: {a(d['dist'])}"
    
    plt.figure(figsize=(10,7))
    plt.title('query_image')
    plt.imshow(getrgb(case_log['query_name']))
    
    plt.figure(figsize=(17,10))
    for i, sample in enumerate(case_log['top_hfnet'][:4]):
        plt.subplot(3, 4, 1 + i)
        plt.imshow(getrgb(sample['db_filename']))
        plt.title(proc_nums(sample))
        
    for i, sample in enumerate(case_log['top_sem'][:4]):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(getrgb(sample['db_filename']))
        plt.title(proc_nums(sample))
        
    for i, sample in enumerate(case_log['top_dist'][:4]):
        plt.subplot(3, 4, 9 + i)
        plt.imshow(getrgb(sample['db_filename']))
        plt.title(proc_nums(sample))
