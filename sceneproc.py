import vsa
import numpy as np


def maskedge(mask):
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    max_y, max_x = mask.shape
    edges = np.zeros((max_y, max_x))
    edges_indx = []
    for y in range(max_y):
        for x in range(max_x):
            if x in [0, max_x - 1] or y in [0, max_y - 1]:
                continue
            if mask[y, x]:
                for dy, dx in deltas:
                    neighbr = mask[y + dy, x + dx]
                    if not neighbr:
                        edges[y + dy, x + dx] = 1
                        edges_indx.append((y + dy, x + dx))
    return edges, edges_indx


def neighbours(mask, semantic_image):
    neigh_names = set()
    _, edges_indx = maskedge(mask)
    for y, x in edges_indx:
        neigh_names.add(semantic_image[y, x])
    return neigh_names


def center_of_mass(mask):
    c = 0
    x_counter, y_counter = 0, 0
    for i in range((mask.shape[0])):
        for j in range((mask.shape[1])):
            if mask[i][j] != 0:
                c += 1
                x_counter += j
                y_counter += i
    return (int(x_counter / c), int(y_counter / c))


def scene_repr(sem_img, mapp, ind2title, skip_wall=True, no_coords=True, depth_img=None):
    elements_in_image = []
    
    for string in sem_img:
        for el in string:
            if not(el in elements_in_image):
                elements_in_image.append(el)
    result = []
    for el in elements_in_image:
        typ = ind2title[mapp[el]]
        if skip_wall and (typ == 'wall' or typ=='ceiling' or typ=='floor'):
            continue
        cur_el = {'name': el}
        cur_el['type'] = typ
        cur_mask = (sem_img == el) * 1
        if not no_coords:
            cm = center_of_mass(cur_mask)
            cur_el['center_of_mass'] = cm
            cur_el['depth'] = depth_img[cm[1]][cm[0]]
        result.append(cur_el)
    return result


def proc_pair(groups, types, pair_names, type):
    for i, group in enumerate(groups):
        if pair_names[0] in group or pair_names[1] in group:
            groups[i].add(pair_names[0])
            groups[i].add(pair_names[1])
            return None
    new_group = {pair_names[0], pair_names[1]}
    groups.append(new_group)
    types.append(type)

            

def get_pairs(sem_img, scene_objs):
    names2scene_objs = {obj['name']: obj for obj in scene_objs}
    pairs = []
    used_names = set()
    seen_names = set()
    groups_with_cmmn_edg = []
    groups_types = []
    for el1 in scene_objs:
        mask1 = sem_img == el1['name']
        seen_names.add(el1['name'])
        neigh_names = neighbours(mask1, sem_img)
        for neigh_name in neigh_names:
            el2 = names2scene_objs.get(neigh_name)
            if el2 is None:
                continue
            if el2['name'] not in seen_names and el1['type'] != el2['type']:
                pair = (el1, el2)
                pairs.append(pair)
                used_names.add(el1['name'])
                used_names.add(el2['name'])
            elif el2['name'] not in seen_names and el1['type'] == el2['type']:
                proc_pair(groups_with_cmmn_edg, groups_types, (el1['name'], el2['name']), el1['type'])
                used_names.add(el1['name'])
                used_names.add(el2['name'])
                
    other_elements = []
    for el in scene_objs:
        if el['name'] not in used_names:
            other_elements.append(el)
    for type in groups_types:
        other_elements.append({'type': type})
    return pairs, other_elements


def create_im(objs):
    item_memory = {}
    item_memory['X'] = vsa.make_unitary(vsa.generate())
    item_memory['Y'] = vsa.make_unitary(vsa.generate())
    item_memory['D'] = vsa.make_unitary(vsa.generate())
    item_memory['R'] = vsa.generate()
    for obj in objs:
        item_memory[obj] = vsa.generate()
    return item_memory


def process_coordinates(center_of_mass, item_memory):
    x = center_of_mass[0] / 13 - 10
    y = center_of_mass[1] / 13 - 10
    X_hd = item_memory['X']
    Y_hd = item_memory['Y']
    X = vsa.power(X_hd, x)
    Y = vsa.power(Y_hd, y)
    res = vsa.bind(X, Y)
    return res


def process_depth(depth, item_memory):
    d = depth 
    D_hd = item_memory['D']
    D = vsa.power(D_hd, d)
    return D


def get_scene_vec(scene, item_memory, use_coord=True):
    cur_vec = np.zeros(len(vsa.generate()))
    for obj in scene:
        if use_coord:
            obj_vec = vsa.bind(item_memory[obj['type']], process_coordinates(obj['center_of_mass'], item_memory))
        else:
            obj_vec = item_memory[obj['type']]
        cur_vec = vsa.bundle(cur_vec, obj_vec) 
    return cur_vec


def edges_scene_vec(item_memory, pairs, other_scene, use_R=True):
    cur_vec = np.zeros(len(vsa.generate()))
    for o1, o2 in pairs:
        if use_R:
            vec1 = vsa.bind(item_memory[o1['type']], item_memory['R'])
            vec2 = vsa.bind(vec1, item_memory[o2['type']])
            cur_vec = vsa.bundle(cur_vec, vec2)
        else:
            vec = vsa.bind(item_memory[o1['type']], item_memory[o2['type']])
            cur_vec = vsa.bundle(cur_vec, vec)
    
    for obj in other_scene:
        obj_vec = item_memory[obj['type']]
        cur_vec = vsa.bundle(cur_vec, obj_vec) 
    return cur_vec


def depth_edges_scene_vec(item_memory, pairs, other_scene, use_R=True):
    cur_vec = np.zeros(len(vsa.generate()))
    for o1, o2 in pairs:
        if use_R:
            depth_vec1 = process_depth(o1['depth'], item_memory)
            vec1 = vsa.bind(item_memory[o1['type']], depth_vec1)
            vec1 = vsa.bind(vec1, item_memory['R'])
            depth_vec2 = process_depth(o2['depth'], item_memory)
            vec2 = vsa.bind(item_memory[o2['type']], depth_vec2)
            vec2 = vsa.bind(vec1, vec2)
            cur_vec = vsa.bundle(cur_vec, vec2)
        else:
            vec = vsa.bind(item_memory[o1['type']], item_memory[o2['type']])
            cur_vec = vsa.bundle(cur_vec, vec)
    
    for obj in other_scene:
        obj_vec = item_memory[obj['type']]
        if 'depth' in obj:
            depth_vec = process_depth(obj['depth'], item_memory)
            obj_vec = vsa.bind(obj_vec, depth_vec)
        cur_vec = vsa.bundle(cur_vec, obj_vec) 
    return cur_vec


def sem2vec(semantic_image, mapping, index_to_title_map, im, use_coord=False, skip_wall=True):
    scene = scene_repr(semantic_image, mapping, index_to_title_map, skip_wall)
    vec = get_scene_vec(scene, im, use_coord)
    return vec


def sem2edgevec(semantic_image, mapping, index_to_title_map, im, use_R=True, skip_wall=True):
    scene = scene_repr(semantic_image, mapping, index_to_title_map, skip_wall)
    pairs, other_scene = get_pairs(semantic_image, scene)
    vec = edges_scene_vec(im, pairs, other_scene, use_R)
    return vec


def sem2depthvec(semantic_image, depth_img, mapping, index_to_title_map, im, use_R=True, skip_wall=False):
    scene = scene_repr(semantic_image, mapping, index_to_title_map, skip_wall, no_coords=False, depth_img=depth_img)
    pairs, other_scene = get_pairs(semantic_image, scene)
    vec = depth_edges_scene_vec(im, pairs, other_scene, use_R)
    return vec