import numpy as np
from scipy.optimize import least_squares
import trimesh
import trimesh.transformations as tr
import openpyscad as ops
import os, sys
# Generate some random points on a quadric surface
# ... (code to generate or load points goes here)
EPS = np.finfo(np.float32).eps
# Define the function to compute the error between a cylinder and the points
# def cylinder_error(params, points):
#     # Unpack the cylinder parameters
#     x0, y0, z0, vx, vy, vz, r, h = params

#     # Compute the distance from each point to the cylinder
#     dx = points[:, 0] - x0
#     dy = points[:, 1] - y0
#     dz = points[:, 2] - z0
#     d = np.sqrt((dy * vz - dz * vy)**2 + (dz * vx - dx * vz)**2 + (dx * vy - dy * vx)**2) - r

#     # Return the error between the cylinder and the points
#     return d

def compute_consistent_plane_frame(normal):
    # Input:  normal is Bx3
    # Returns: x_axis, y_axis, both of dimension Bx3
    #print('normal ', normal)
    batch_size = normal.shape[0]
    candidate_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) # Actually, 2 should be enough. This may still cause singularity TODO!!!
    y_axes = []
    for tmp_axis in candidate_axes:
        np_axis = np.tile(tmp_axis, (batch_size, 1)) # Bx3
        # print('normal ', normal.shape)
        # t = tmp_axis.reshape(1, 3)
        # print('tmp_axis ', t.shape)
        y_axes.append(np.cross(normal, np_axis))
    y_axes = np.stack(y_axes, axis=0) # QxBx3
    #print('y_axes ', y_axes.shape)
    y_axes_norm = np.linalg.norm(y_axes, axis=2) # QxB
    # choose the axis with largest norm
    y_axes_chosen_idx = np.argmax(y_axes_norm, axis=0) # B
    # y_axes_chosen[b, :] = y_axes[y_axes_chosen_idx[b], b, :]
    # indices_0 = np.tile(np.expand_dims(y_axes_chosen_idx, axis=1), (1, 3)) # Bx3
    # indices_1 = np.tile(np.expand_dims(np.arange(batch_size), axis=1), (1, 3)) # Bx3
    # indices_2 = np.tile(np.expand_dims(np.arange(3), axis=0), (batch_size, 1)) # Bx3
    # indices = np.stack([indices_0, indices_1, indices_2], axis=2) # Bx3x3
    # print('indices_0 ', indices_0)
    # print('indices_1 ', indices_1)
    # print('indices_2 ', indices_2)
    y_axes = y_axes[y_axes_chosen_idx].reshape(batch_size, 3)
    #np.take_along_axis(y_axes, indices, axis=0) # Bx3
    #print('y_axes ', y_axes.shape)
    y_axes = y_axes / (np.linalg.norm(y_axes, axis=1, keepdims=True) + EPS)
    x_axes = np.cross(y_axes, normal) # Bx3

    return x_axes, y_axes

# def fit_sphere_numpy(points, normals):
#     weights = np.ones((points.shape[0], 1))
#     dimension = points.shape[1]
#     N = weights.shape[0]
#     sum_weights = np.sum(weights)
#     A = 2 * (- points + np.sum(points * weights, 0) / sum_weights)
#     dot_points = np.sum(points * points, 1)
#     normalization = np.sum(dot_points * weights.reshape(-1)) / sum_weights
#     # print(np.sum(dot_points * weights))
#     # print('normalization, ', normalization)
#     # print('dot_points, ', dot_points)
#     # print('dot_points max ', np.amax(dot_points))
#     # print('dot_points min ', np.amin(dot_points))
#     Y = dot_points - normalization
#     Y = Y.reshape((N, 1))
#     # print('A, ', A)
#     # print('Y, ', Y)
#     A = weights * A
#     Y = weights * Y
    
#     center = -np.linalg.lstsq(A, Y, rcond=-1)[0].reshape((1, dimension))
#     radius = np.sqrt(np.sum(weights[:, 0] * np.sum((points - center) ** 2, 1)) / sum_weights)
#     return center, radius

def fit_sphere_numpy(points):
    print('fitting sphere...')
    weights = np.ones((points.shape[0], 1))
    dimension = points.shape[1]
    N = weights.shape[0]
    sum_weights = np.sum(weights)
    A = 2 * (- points + np.sum(points * weights, 0) / sum_weights)
    dot_points = np.sum(points * points, 1)
    normalization = np.sum(dot_points * weights.reshape(-1)) / sum_weights
    # print(np.sum(dot_points * weights))
    # print('normalization, ', normalization)
    # print('dot_points, ', dot_points)
    # print('dot_points max ', np.amax(dot_points))
    # print('dot_points min ', np.amin(dot_points))
    Y = dot_points - normalization
    Y = Y.reshape((N, 1))
    # print('A, ', A)
    # print('Y, ', Y)
    A = weights * A
    Y = weights * Y
    
    center = -np.linalg.lstsq(A, Y, rcond=-1)[0].reshape((1, dimension))
    radius = np.sqrt(np.sum(weights[:, 0] * np.sum((points - center) ** 2, 1)) / sum_weights)
    return center, radius

def fit_cylinder_numpy(points, normals):
    print('fitting cylinder...')
    n_points = points.shape[0]

    _, s, V = np.linalg.svd(normals, compute_uv=True)
    a = V.T[:, np.argmin(s)]
    a = np.reshape(a, (1, 3))

    # find the projection onto a plane perpendicular to the axis
    a = a.reshape((1, 3))
    a = a / (np.linalg.norm(a, ord=2) + EPS)
    
    # prj_circle = points - ((points @ a).T * a).T
    # print(prj_circle)
    x_axes, y_axes = compute_consistent_plane_frame(a)

    x_coord = np.sum(points * x_axes, axis=1) # BxKxN
    y_coord = np.sum(points * y_axes, axis=1) # BxKxN
    P_proj = np.stack([x_coord, y_coord], axis=1) # BxKxNx2, 2D projection point
    P_proj_reshaped = np.reshape(P_proj, [n_points, 2]) # BKxNx2

    center, radius = fit_sphere_numpy(P_proj_reshaped)
    new_center = center[0, 0] * x_axes + center[0, 1] * y_axes

    diff = points - new_center.reshape(1, 3)
    #print('radius, ', radius)
    distance = diff @ a.transpose(1, 0)
    height = np.max(distance) - np.min(distance)
    new_center = new_center + (np.max(distance) + np.min(distance))/2 * a
    new_center = new_center.reshape(-1)
    return a, new_center, radius, height

def visualize_cylinder(direction, center, radius, height):
    # print('direction ', direction)
    # print('center ', center)
    # print('radius ', radius)
    # print('height ', height)
    a = np.array([0, 0, 1])
    b = direction.reshape(-1) #np.array([vx, vy, vz])

    axis = np.cross(a, b)
    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    axis = axis / np.linalg.norm(axis)
    print('angle, ', angle)
    print('axis, ', axis)
    R = tr.rotation_matrix(angle, axis)

    euler_angles = tr.euler_from_matrix(R, 'rxyz')

    T = tr.translation_matrix(center.reshape(-1))

    S = np.diag([radius, radius, 1, 1])

    # Combine matrices to form the final transformation matrix
    M = np.dot(T, np.dot(R, S))

    cylinder_mesh = trimesh.primitives.Cylinder(radius=1.0, height=height, sections=32, transform=M)
    cylinder_mesh.export('cylinder.obj')
    return cylinder_mesh, euler_angles, angle, axis

def visualize_cone(radius, height, direction, apex, angle):
    # print('radius ', radius)
    # print('height ', height)
    # print('direction ', direction)
    # print('apex ', apex)
    # print('angle ', angle)

    a = np.array([0, 0, -1])
    b = direction.reshape(-1) #np.array([vx, vy, vz])

    axis = np.cross(a, b)
    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    axis = axis / np.linalg.norm(axis)

    R = tr.rotation_matrix(angle, axis)
    euler_angles = tr.euler_from_matrix(R, 'rxyz')

    T = tr.translation_matrix(apex.reshape(-1) + height*direction.reshape(-1))

    # R = trimesh.transformations.rotation_matrix(axis, angle)

    S = np.diag([1, 1, 1, 1])

    # Combine matrices to form the final transformation matrix
    M = np.dot(T, np.dot(R, S))

    cone_mesh = trimesh.creation.cone(radius, height, sections=32, transform=M)
    cone_mesh.export('cone.obj')
    return cone_mesh, euler_angles, apex.reshape(-1) + height*direction.reshape(-1), angle, axis

def visualize_sphere(center, radius):
    sphere = trimesh.creation.icosphere(radius=radius, subdivisions=3)
    sphere.apply_translation(center.reshape(-1))
    #sphere.export('sphere.obj')
    return sphere

def visualize_plane(normal, d):
    normal = normal.reshape(-1)
    size = 1.5
    a = np.array([0, 0, 1])
    b = normal #np.array([vx, vy, vz])

    axis = np.cross(a, b)
    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    axis = axis / np.linalg.norm(axis)
    R = tr.rotation_matrix(angle, axis)
    euler_angles = tr.euler_from_matrix(R, 'rxyz')

    print('angle, ', angle)
    print('axis, ', axis)
    #print('normal ', normal)
    trans = size/2 - (d/np.linalg.norm(normal))
    print('d, ', (d/np.linalg.norm(normal)))
    print('trans, ', trans)
    T = tr.translation_matrix(trans*normal)

    S = np.diag([1, 1, 1, 1])
    M = np.dot(T, np.dot(R, S))
    #print('M, ', M)
    plane_mesh = trimesh.creation.box(extents=np.array([size, size, size]), transform=M)
    #print(plane_mesh.extents)
    plane_mesh.export('plane.obj')
    #return plane_mesh
    return size, euler_angles, trans*normal, angle, axis

def fit_plane_numpy(points):
    """
    Fits plane
    :param points: points with size N x 3
    :param weights: weights with size N x 1
    """
    print('fitting plane...')
    weights = np.ones((points.shape[0], 1))
    X = points - np.sum(weights * points, 0).reshape((1, 3)) / np.sum(weights, 0)
    _, s, V = np.linalg.svd(weights * X, compute_uv=True)
    a = V.T[:, np.argmin(s)]
    a = np.reshape(a, (1, 3))
    a = a / np.linalg.norm(a)

    d = - np.sum(weights * (a @ points.T).T) / np.sum(weights, 0)
    return a, d
        
def fit_box_numpy(mesh):
    print('fitting box...')
    mesh_sizes = mesh.bounding_box_oriented.primitive.extents
    #print(mesh_sizes)
    transform = mesh.bounding_box_oriented.primitive.transform
    # Decompose the matrix into its scale, shear, angles, translation, and perspective components
    scale, shear, angles, translate, perspective = trimesh.transformations.decompose_matrix(transform)
    # print('mesh_sizes, ', mesh_sizes)
    # print('scale, ', scale)
    # print('angles, ', angles)
    # print('shear, ', shear)
    # print('perspective, ', perspective)
    # Extract the rotation angles from the angles part of the decomposition
    rot_x, rot_y, rot_z = angles
    
    # Convert the rotation angles to a rotation matrix
    rotation_matrix = trimesh.transformations.euler_matrix(rot_x, rot_y, rot_z)

    # Extract the rotation axis and angle from the rotation matrix
    rotation_angle, rotation_axis, _ = trimesh.transformations.rotation_from_matrix(rotation_matrix)
    # print('rotation_angle, ', rotation_angle)
    # print('rotation_axis, ', rotation_axis)
    # print('translate, ', translate)
    R = rotation_matrix
    T = tr.translation_matrix(translate)

    S = np.diag([1, 1, 1, 1])
    M = np.dot(T, np.dot(R, S))

    mesh.bounding_box_oriented.export('box.obj')
    plane_mesh = trimesh.creation.box(extents=mesh_sizes, transform=M)
    plane_mesh.export('box2.obj')
    return mesh.bounding_box_oriented, mesh_sizes, angles, translate, rotation_angle, rotation_axis


def fit_cone_numpy(points, normals):
    """ Need to incorporate the cholesky decomposition based
    least square fitting because it is stable and faster."""
    print('fitting cone...')
    weights = np.ones((points.shape[0], 1))
    N = points.shape[0]
    A = weights * normals
    Y = np.sum(normals * points, axis=1).reshape((N, 1))
    Y = weights * Y

    # if condition number is too large, return a very zero cone.
    if np.linalg.cond(A) > 1e5:
        if show_warning:
            print("condition number is large, cone")
            print(np.sum(normals), np.sum(points), np.sum(weights))
        return np.zeros(1)+1, np.zeros(1)+1, np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.zeros(1)+1

    c = np.linalg.lstsq(A, Y, rcond=1e-3)[0]

    a, _ = fit_plane_numpy(normals)
    if np.sum(normals @ a.transpose(1, 0)) > 0:
        # we want normals to be pointing outside and axis to
        # be pointing inside the cone.
        a = - 1 * a
    #print('a, ', a.shape)
    diff = points - c.transpose(1, 0)
    radius = np.cross(diff, a)
    radius = np.linalg.norm(radius, axis=1)
    radius = np.max(radius)
    #print('radius, ', radius)
    height = diff @ a.transpose(1, 0)
    height = np.max(height)
    #print('height, ', height)

    diff = diff / np.linalg.norm(diff, axis=1)[:, np.newaxis]
    diff = diff @ a.transpose(1, 0)

    # This is done to avoid the numerical issue when diff = 1 or -1
    # the derivative of acos becomes inf
    diff = np.abs(diff)
    diff = np.clip(diff, a_min=None, a_max=0.999)
    theta = np.sum(weights * np.arccos(diff)) / (np.sum(weights) + np.finfo(np.float32).eps)
    theta = np.clip(theta, a_min=1e-3, a_max=np.pi / 2 - 1e-3)
    
    return radius, height, c, a, theta

csg_dir = 'D:\\workspace\\DualCSG\\dualcsg\\Reconstructions_test_csg_primi_all_3'
name_file = 'test_names.npz'
npz_shapes = np.load(name_file)
data_names = npz_shapes['test_names']
shape_index = 252

csg_file = os.path.join(csg_dir, data_names[shape_index]+'_stage_3', 'csg.npz')
op_data = np.load(csg_file)
left_ops = op_data['left_ops']
right_ops = op_data['right_ops']
convex_mask = op_data['unused_convex']

p_dim = left_ops.shape[0]*2
c_dim = left_ops.shape[1]*2

primi_dir = os.path.join(csg_dir, data_names[shape_index]+'_stage_3', 'primi_for_render')
primi_names = os.listdir(primi_dir)      
primi_names.sort()
primitive_list = []
primi_indicator = np.zeros((len(primi_names)))
plane_indicator = np.zeros((len(primi_names)))

print(f'split index {int(p_dim/4)}, {int(p_dim/2)}, {int(p_dim*3/4)}')

for ind, obj in enumerate(primi_names):
    primi_indicator[ind] = int(obj[:-4])
    print(f'processing primi {primi_indicator[ind]}')
    mesh = trimesh.load(os.path.join(primi_dir, obj))

    points, face_inds = mesh.sample(10000, return_index=True)
    #samps, face_inds = shape_mesh.sample(sample_num, return_index=True)
    normals = mesh.face_normals[face_inds]

    mesh_sizes = mesh.bounding_box_oriented.primitive.extents

    if np.min(mesh_sizes) < 0.01:
        normal, dis = fit_plane_numpy(points)
        print('normal, ', normal)
        print('dis, ', dis)
        #normal*(x,y,z) -d =0
        #the normal should be at the oppositive direction for cube meshing
        if np.sum(normal @ normals[0:1, :].transpose(1, 0)) > 0:
            normal = - 1 * normal
            dis = - dis
        print('best fit primitive: plane')
        print('normal, ', normal)
        print('dis, ', dis)
        size, euler_angles, translation, angle, axis = visualize_plane(normal, dis)
        primitive_list.append(ops.Cube([size, size, size], center=True).rotate(180*angle/np.pi, v = axis.tolist()).translate([translation[0], translation[1], translation[2]]))
        plane_indicator[ind] = 1
    else:
        if (primi_indicator[ind] >= int(p_dim/4) and primi_indicator[ind] < int(p_dim/2)) or \
        (primi_indicator[ind] >= int(p_dim*3/4)):
            normals = normals * -1
            print('inverse primi')
            
        #fit_box_numpy(mesh)
        #fit other type of primitives
        box_mesh, mesh_sizes, angles, translation, rotation_angle, rotation_axis = fit_box_numpy(mesh)
        _, dis, _ = trimesh.proximity.closest_point(box_mesh, points)
        ave_dis_box = np.mean(dis)

        s_center, s_radius = fit_sphere_numpy(points)
        sphere_mesh = visualize_sphere(s_center, s_radius)
        _, dis, _ = trimesh.proximity.closest_point(sphere_mesh, points)
        ave_dis_sphere = np.mean(dis)

        cy_direction, cy_center, cy_radius, cy_height = fit_cylinder_numpy(points, normals)
        cylinder_mesh, cy_euler, cy_angle, cy_axis = visualize_cylinder(cy_direction, cy_center, cy_radius, cy_height)
        _, dis, _ = trimesh.proximity.closest_point(cylinder_mesh, points)
        ave_dis_cylinder = np.mean(dis)

        co_radius, co_height, co_apex, co_direction, co_theta = fit_cone_numpy(points, normals)
        cone_mesh, co_euler, co_trans, co_angle, co_axis = visualize_cone(co_radius, co_height, co_direction, co_apex, co_theta)
        _, dis, _ = trimesh.proximity.closest_point(cone_mesh, points)
        ave_dis_cone = np.mean(dis)

        all_dis = np.array([ave_dis_box, ave_dis_sphere, ave_dis_cylinder, ave_dis_cone])
        print('all_dis, ', all_dis)
        min_index = np.argmin(all_dis)

        # Print the result
        print(min_index)
        if min_index == 0:
            print('best fit primitive: box')
            primitive_list.append(ops.Cube(mesh_sizes.tolist(), center=True).rotate(rotation_angle*180/np.pi, v=rotation_axis.tolist()).translate([translation[0], translation[1], translation[2]]))
        elif min_index == 1:
            print('best fit primitive: sphere')
            #print('s_center ', s_center)
            primitive_list.append(ops.Sphere(r=s_radius, _fn=100).translate([s_center[0, 0], s_center[0, 1], s_center[0, 2]]))
        
        elif min_index == 2:
            print('best fit primitive: cylinder')
            # print(cy_euler)
            # print(cy_center)
            primitive_list.append(ops.Cylinder(h=cy_height, r=cy_radius, _fn=100, center=True).rotate(cy_angle*180/np.pi, v=cy_axis.tolist()).translate([cy_center[0], cy_center[1], cy_center[2]]))
        
        elif min_index == 3:
            print('best fit primitive: cone')
            primitive_list.append(ops.Cylinder(h=co_height, r1=co_radius, r2=0, _fn=100, center='false').rotate(co_angle*180/np.pi, v=co_axis.tolist()).translate([co_trans[0], co_trans[1], co_trans[2]]))
        
        else:
            print('error')
    # except:
    #     normal, dis = fit_plane_numpy(points)
    #     print('normal, ', normal)
    #     print('dis, ', dis)
    #output a numpy file
# for i, primi in enumerate(primitive_list):
#     primi.write(f"{i}.scad")
#fitting cylinder
print('primi_indicator, ', primi_indicator)
union_nodes_left = ops.Union()
union_nodes_right = ops.Union()
basic_cube = ops.Cube([1, 1, 1], center=True)

for i in range(c_dim):
    if convex_mask[i] == 0:
        int_node = ops.Intersection()
        for j in range(p_dim):
            if (i<c_dim/2 and j<p_dim/2 and left_ops[j,i]>0) or \
                    (i>=c_dim/2 and j>=p_dim/2 and right_ops[j-int(p_dim/2),i-int(c_dim/2)]>0):

                indices = np.where(primi_indicator == j)[0]
                # print('indices, ', indices)
                # print('indices, ', indices[0].shape)
                if plane_indicator[indices[0]] == 1:
                    int_node.append(primitive_list[indices[0]])
                elif (j >= int(p_dim/4) and j < int(p_dim/2)) or \
                    (j >= int(p_dim*3/4)):
                        differ_node = ops.Difference()
                        differ_node.append(basic_cube)
                        differ_node.append(primitive_list[indices[0]])
                        int_node.append(differ_node)
                else:
                    int_node.append(primitive_list[indices[0]])

        if i < c_dim/2:
            union_nodes_left.append(int_node)
        else:
            union_nodes_right.append(int_node)

differ_node = ops.Difference()
differ_node.append(union_nodes_left)
differ_node.append(union_nodes_right)
out_file = os.path.join(csg_dir, data_names[shape_index]+'_stage_3', 'final.scad')
differ_node.write(out_file)

#exit(0)
# x0, y0, z0, vx, vy, vz, r, h = params
# x0, y0, z0, vx, vy, vz, r, h = 1, 1, 1, 1, 1, 0, 1, 3
#----------------------------
