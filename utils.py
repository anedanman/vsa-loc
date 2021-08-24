from pathlib import Path
import h5py
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import vsa
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import ast 
from sceneproc import create_im, sem2vec, sem2edgevec


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


#{q: query name, top5hfnet: [{'filename':, 'hfnetsim':, 'semsim':, 'dist':}, ], top5dist: [...], top5sem:}
def proccess_descr_adv(path_to_hdf_files, path_to_global_descriptors, query_image, database, sem_path, k=0.5, mode='', n=5):
    sims = []
    result = []
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
        dist = get_dist(query_image, db_image, path_to_hdf_files)
        result.append({'db_filename': db_image,
                       'hfnetsim': sim1,
                       'semsim': sim2,
                       'dist': dist})
        if mode == 'semantic':
            sim = sim2
        if mode == 'default':
            sim = sim1
        sims.append(sim)
    tophfnet = sorted(result, key=lambda x: -x['hfnetsim'])
    topdist = sorted(result, key=lambda x: x['dist'])
    topsem = sorted(result, key=lambda x: -x['semsim'])   
        
    num_new = np.argmax(sims)
    db_image = database[num_new]
    simil_score = sims[num_new]
    return simil_score, db_image, tophfnet[:n], topdist[:n], topsem[:n]


def adv_test(path_to_hdf='./datasets/Habitat/HPointLoc/',
              path_to_hfnet='./vectors/HF-Net_descriptors',
              path_to_sem='./vectors/semantic_edges',
              mode='default', k=0.9, dist_thr=20):
    results = []
    problem_cases = []
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
            simil_score, db_image, tophfnet, topdist, topsem = proccess_descr_adv(os.path.join(path_to_hdf, map_name),
                                                                              os.path.join(path_to_hfnet, map_name),
                                                                              query, database,
                                                                              os.path.join(path_to_sem, map_name),
                                                                              mode=mode, k=k)
            dist_diff = get_dist(query, db_image, os.path.join(path_to_hdf, map_name))
            for d in distances:
                if dist_diff < d:
                    res[d] += 1
            if dist_diff > dist_thr:
                problem_cases.append({'query_name': query, 'top_hfnet': tophfnet,
                                    'top_dist': topdist, 'top_sem': topsem})
        for i in res:
            res[i] = res[i] / len(queries)
        results.append(res)
        
    answer = {i:0 for i in distances}
    for res in results:
        for i in res:
            answer[i] += res[i]
    for i in answer:
        answer[i] = answer[i] / len(results)
    return answer, problem_cases
    
    
    
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
    
    index_to_title_map_query = ast.literal_eval(str(np.array(hdf5_file_query1['index_to_title_map'])))
    mapping_query = np.array(hdf5_file_query1['mapping'])
        
    return pose_44_query1, semantic, index_to_title_map_query, mapping_query


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
        _, db_sem, index_to_title_map, mapping = get_pose(name, path_to_hdf5_datasets)
        if im is None:
            im = create_im(index_to_title_map.values())
            
        db_vec = sem2vec(db_sem, mapping, index_to_title_map, im)
        with open(os.path.join(sem_path, name), 'wb') as f:
            np.save(f, db_vec)
            
            
def save_edge_vectrs(descrs_path, sem_path, path_to_hdf5_datasets, use_R=True, skip_wall=True):
    im = None
    for name in tqdm(os.listdir(descrs_path)):
        _, db_sem, index_to_title_map, mapping = get_pose(name, path_to_hdf5_datasets)
        if im is None:
            im = create_im(index_to_title_map.values())
            
        db_vec = sem2edgevec(db_sem, mapping, index_to_title_map, im, use_R, skip_wall)
        with open(os.path.join(sem_path, name), 'wb') as f:
            np.save(f, db_vec)
            
            
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
    query_pose, _, _, _ = get_pose(query_img, path_to_hdf_files)
    db_pose, _, _, _ = get_pose(database_img, path_to_hdf_files)
    dist_diff, _ = poses_diff(query_pose, db_pose)
    return dist_diff
