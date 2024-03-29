import bpy
import bmesh
import random
import sys
from math import sqrt,pi,sin,cos,acos,asin
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm
#from np.linalg import norm
from julia import Main#import julia
import mathutils
from functools import reduce


#############################
### Functions for Logging ###
#############################

VERBOSITY = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
log_file = open('log.txt','w+')

def logger(func):
    fname = func.__name__
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if VERBOSITY:
            print("{0}({1},{2}) =>\n{3}".format(fname, args if args else "", kwargs if kwargs else "", output))
        return output
    return wrapper

def printl(*args):
    line = ' '.join([str(a) for a in args])
    log_file.write(line+'\n')
    print(line)


#############################
###   Helper Functions    ###
#############################

def append_material(obj, mat):
    if VERBOSITY:
        print('adding ',mat.name,' to ', obj.name)
    obj.data.materials.append(mat)

def angle_offset(vec1,vec2):
    cosang = vec1.dot(vec2)/(vec1.length*sqrt(sum([x**2 for x in vec2])))
    return acos(cosang)

def vector_sum(vecs):
    '''
    Built-in sum() doesnt handle objects which havent implemented __radd__ since it starts by computing `0+vecs[0]`
    '''
    return reduce((lambda x,y:x+y),vecs)

def init_material(color, name):
    new_mat = bpy.data.materials.new(name=name)
    if len(color) == 3:
        new_mat.diffuse_color = tuple(list(color)+[1])
    return new_mat

@logger
def get_side_curve():
    archetype = random.getrandbits(1)
    if archetype:
        # Broad-shouldered archetype
        shoulder = 1.5 + random.random()
        waist = 1 + .4*random.random()
        bodylen = random.randint(8,15)
        
        # if shoulder/waist curve is linear, quadratic, or cubic
        shoulder_curvature = random.choice([1,2,3])
        waist_curvature = random.choice([1,2,3])
        
        #.35 and .3 constants are someone arbitrary, tuned to look nicely proportional
        # shoulder goes from `shoulder` to `shoulder+.35`
        body = [shoulder + x**(1/shoulder_curvature) for x in np.linspace(0,.35**shoulder_curvature,4)]
        # torso is just linear between shoulder and waist
        body = body + [x for x in np.linspace(shoulder+.35,waist+.3,bodylen)]
        # waist goes from `waist+.3` to `waist`
        body = body + [waist+x**(1/waist_curvature) for x in np.linspace(.3**waist_curvature,0,3)]

    else:
        # Pot-bellied archetype
        shoulder = 1 + .5*random.random()
        waist = random.random()
        bodylen = random.randint(8,15)
        waist_offset = random.randint(-2,2)
        
        # if shoulder/waist curve is linear, quadratic, or cubic
        upper_curvature = random.choice([1,2,3])
        lower_curvature = random.choice([1,2,3])
        
        # upper half goes from `shoulder` to `shoulder+waist`
        body = [shoulder + x**(1/upper_curvature) for x in np.linspace(0,waist**upper_curvature, bodylen//2 - waist_offset)]
        # lower half goes from `shoulder+waist` to `shoulder`
        body = body + [shoulder + x**(1/lower_curvature) for x in np.linspace(waist**lower_curvature, 0, bodylen//2 + waist_offset)]
    
    feet = []
    yr = body + feet
    return yr 

def eye_position(cx,cy,cz,rad,n1,n2,eye_rad):
    minwidth = asin(eye_rad/rad)
    width = random.random()*(pi/2-.4 - minwidth) + minwidth # width in radians between [min,pi/2]
    height = .25*pi*(1+random.random()) # incline angle (0 is straight up, pi straight down) on [.25pi,.5pi]
    c = cos(width); s = sin(width); ss = sin(height)
    # just a spherical to cartesian conversion in forward basis
    x1 = -rad*(n1*c - n2*s)*ss
    y1 = rad*(n2*c - n1*s)*ss
    z =  rad*cos(height)
    
    # householder reflection in xy plane
    m1 = n2; m2 = -n1
    p1dotn = x1*m1 + y1*m2
    x2 = x1-2*p1dotn*m1
    y2 = y1-2*p1dotn*m2

    return (cx+x1,cy+y1,cz+z),(cx+x2,cy+y2,cz+z)




    

def test_new_point(xy_bounds, circles, new_x, new_y, new_rad):
    for px,py,r in circles:
        d = sqrt((px - new_x)**2 +(py - new_y)**2)
        if d <= r+new_rad:
            return False
    return True

def sample_position(xy_bounds, circles, new_rad, max_tries = 100):
    is_valid = False
    tries = 0
    while (not is_valid) and tries < max_tries:
        x = random.random()
        y = random.random()
        linscale = lambda z,b: b[0] + z*(b[1]-b[0])
        x = linscale(x,xy_bounds[0])
        y = linscale(y,xy_bounds[1])
        is_valid = test_new_point(xy_bounds,circles,x,y,new_rad)
        tries += 1
    if tries == max_tries:
        print("Iceberg was overstuffed.  One penguin fell into the ocean.")
    return x,y
 
#############################
###   Useful Generators   ###
#############################

def side_faces(bm):
    f_indices = [f.index for f in bm.faces]
    for i_f in f_indices:
        bm.faces.ensure_lookup_table()
        f = bm.faces[i_f]
        f_direction = f.normal
        if f_direction.dot([0,0,1]) != 0:
            continue
        yield f

def vertical_edges(f):
        e_indices = [e.index for e in f.edges]
        edges = frozenset([e for e in f.edges])
        for e in edges:
            e_direction = e.verts[1].co - e.verts[0].co
            # ignore top edges
            if e_direction.dot([0,0,1]) == 0 or not (e_direction[0] == 0 and e_direction[1] == 0):
                continue
            maxv = e.verts[1]
            if e_direction[-1] < 0:
                maxv = e.verts[0]
            if VERBOSITY:
                printl("returning edge direction {0} with verts: \n{1} and \n{2}".format(e_direction, maxv.co.xy,e.other_vert(maxv).co))
            yield e,maxv


#############################
###  Top-level Functions  ###
#############################
 
def shape_body(bm,smoothness=24,iters=100):
    ycoeffs = get_side_curve()
    
    #TODO: Think about doing this in a vectorized fashion so that we need only one Julia call.  May be much faster.
    knots,fvals = Main.poly_get_knots(ycoeffs,smoothness,iters)

    # simple test values to use in debugging where body shape is not necessary and want to comment out Julia call
    #knots = [0,.2,.4,.7,1]
    #fvals = [.9,1.1,1.5,1]
    knots = knots[1:-1]
    fvals = fvals[1:-1]
    knots = [1-k for k in knots[::-1]]
    fvals = fvals[::-1]
    return extrude_cylinder_side(bm,knots,fvals)

def colorize(obj,mat=None):
    #activeObject = bpy.context.active_object #Set active object to variable

    #mat.diffuse_color = (1, 5, .2, 1) #yellow
    probs = [100,10,80,100,10,100,100,200,10,20]
    colors = [
            (123,104,238),  # medium slate blue
            (230,230,250),  # lavender
            (0,191,255),    # deep sky blue
            (30,144,255),   # dodger blue
            (100,149,237),  # corn flower blue
            (70,130,180),   # steel blue
            (0,0,205),      # medium blue
            (72,61,139),    # dark slate blue
            (112,128,144),  # slate gray
            (192,192,192),  # silver
            ]
    if not mat:
        color = random.choices(colors, probs)[0]
        color = [c + random.uniform(-10,10) for c in color]
        color_scaled = [c/255 for c in color]
        mat = init_material(color_scaled, "skin")    
    
    append_material(obj, mat)
    #mat.diffuse_color = color_scaled #blue
    #return color_scaled
    return mat

def colorize_chest(obj, chest_mat, epsilon = pi/6):
    append_material(obj,chest_mat)
    for i,f in enumerate(obj.data.polygons):
        if abs(angle_offset(f.normal,[0,1,0])) < epsilon:
            f.material_index += 1

def colorize_beak(obj,mesh,beak_mat,beak_faces):
    append_material(obj,beak_mat)
    obj.data.materials.append(beak_mat)

    n = len(obj.data.materials)
    fs = [f for f in mesh.faces if beak_faces in [v.index for v in f.verts]]
    for f in fs:
        f.material_index = n-1

@logger
def extrude_cylinder_side(bm,knots,fvals):
    '''
    knots should be a list of n values in (0,1) with 0 being top of body, 1 being the bottom
    fvals are the extent of protusions, in units of multiples of the cylinder's radius
    '''
    maxr = max(fvals)
    vertice_lvls = dict()
    for e,v in vertical_edges(bm):
        prevknot = 0
        for knot in knots:
            knot_scaled = (knot-prevknot)/(1-prevknot)
            prevknot = knot

            v = min(e.verts, key=lambda v: v.co[2])
            enew,vnew = bmesh.utils.edge_split(e,v,knot_scaled)
            if knot in vertice_lvls:
                vertice_lvls[knot].append(vnew)
            else:
                vertice_lvls[knot] = [vnew]
        
    for f in side_faces(bm):
        for k_i,knot in enumerate(knots):
            vertices = vertice_lvls[knot]
            N = len(vertices)
            
            # TODO: fix this disgusting hack to map vertex pairs back to face
            # find index in vertices to apply
            if k_i == 0:
                for i,v in enumerate(vertices):
                    if v in f.verts and vertices[(i+1)%N] in f.verts:
                        start = i
                        break
                curr = start
                nextcurr = (curr+1)%N
            ff = bmesh.utils.face_split(f,vertices[curr],vertices[nextcurr])
            ff[0].material_index = f.material_index
            
    for knot,fval in zip(knots,fvals):
        for v in vertice_lvls[knot]:
            v.co[0:2] = fval*v.co.xy
    return maxr

@logger
def add_beak(bm):
    v = max(bm.verts, key=lambda x: x.co[1])
    v.co[1] += random.normalvariate(5,1)
    return v.index

def get_bm():        
    mesh = bpy.context.object.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    return bm,mesh
    
def rot_x(bm,angle=pi/6):
    ct = cos(angle); st = sin(angle)
    rotmat = mathutils.Matrix([[1,0,0,0],[0,ct,-st,0],[0,st,ct,0],[0,0,0,1]])
    bm.transform(rotmat)

def rot_y(bm,angle=pi/6):
    ct = cos(angle); st = sin(angle)
    rotmat = mathutils.Matrix([[ct,0,st,0],[0,1,0,0],[-st,0,ct,0],[0,0,0,1]])
    bm.transform(rotmat)

def add_wing(offset=5,tilt=pi/24):
    bpy.ops.mesh.primitive_uv_sphere_add(location=(x+offset,y,cyl_scale*3/4),radius=rad)
    obj = bpy.context.active_object
    bm,mesh = get_bm()
    mat = mathutils.Matrix([[.2,0,0,0],[0,1,0,0],[0,0,2,0],[0,0,0,0]])
    bm.transform(mat)
    if offset > 0:
        angle = -tilt
    else:
        angle = tilt
    rot_y(bm,angle)
    bm.to_mesh(mesh)
    bm.free()
    #print("wing: ", obj)
    return obj

def add_eye(e,c,r,skew_factor=.2):
    #print(list(c),list(e))
    matrix = Main.transformation_matrix(list(c), list(e), skew_factor)
    matrix = mathutils.Matrix(matrix)
    #print(matrix)
    bpy.ops.mesh.primitive_uv_sphere_add(location=e,radius=r)
    eye = bpy.context.active_object
    bm,mesh = get_bm()
    bm.transform(matrix)
    bm.to_mesh(mesh)
    bm.free()
    return eye        

def add_sky(world):
    # Add sky texture to world node tree
    node_tree = world.node_tree
    node = node_tree.nodes.new("ShaderNodeTexSky")
    node.select = True
    node_tree.nodes.active = node
    # Link sky texture to world output
    out = node.outputs[0]
    bg = node_tree.nodes["World Output"]
    node_tree.links.new(out, bg.inputs[0])

def add_water(sea_level, filepath="/Users/Bryan/Documents/Projects/Blender/Madame-Julies-Penguins/water_surface_texture.jpg"):
    # initialize water
    sz = 500
    bpy.ops.mesh.primitive_plane_add(size=sz, location=(0,0,sea_level))
    water_obj = bpy.context.active_object

    water_mat = init_material([],"water")
    water_mat.use_nodes = True
    node_tree = water_mat.node_tree
    bsdf = node_tree.nodes["Principled BSDF"]
    img = bpy.data.images.load(filepath)
    node = node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    node_tree.links.new(bsdf.inputs['Base Color'], node.outputs['Color'])
    node.select = True
    node_tree.nodes.active = node

    colorize(water_obj, water_mat)
    return water_obj

def add_iceberg(height=5):
    # initialize iceberg
    iceberg_obj = bpy.data.objects['Cube']
    iceberg_obj.scale = (50,50,height/2)
    iceberg_obj.location = (0,0,-height/2)
    colorize(iceberg_obj, ice_mat)
    iceberg_bounds = ((-50,50),(-50,50))
    return iceberg_bounds

# eyes = {'slanted', 'circular'}
# beak = {'short/stubby', 'long narrow'}
# body = {'macho', 'chubby'}
#


if __name__ == '__main__':

    
    # Initialize Julia files
    Main.include('interp.jl')
    Main.include('geometry.jl')
    
    # Initialize blender settings/objects
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)
    world = bpy.data.worlds["World"]
    camera_obj = bpy.data.objects["Camera"]

    # Number of penguins
    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 0

    # Initialize camera position
    camera_obj.location = (0,250,125)
    camera_obj.rotation_euler = (4.276,pi,0)
    camera_obj.data.clip_end = 1000
    

    avgx = 0
    avgy = 0
    
    # Initialize materials 
    chest_mat = init_material([1,1,1],"chest")
    beak_mat = init_material([1,140/255,0],"beak")
    ice_mat = init_material([1,1,1],"ice")
    eye_mat = init_material([0,0,.1],"eyes")
    
    
    # Add environment
    height = 5
    iceberg_bounds = add_iceberg(5)
    add_water(-height)
    add_sky(world)

 
    positions = []
    
    for i in range(N):
        
        # Choose height and core radius
        cyl_scale = 2*random.random() + 6
        rad = random.normalvariate(1,.1) + cyl_scale*.5 - .5

        # Add body
        bpy.ops.mesh.primitive_cylinder_add(location=(0,0,0),radius=rad)
        body_obj = bpy.context.active_object
        body_obj.scale = (1,1,cyl_scale)
        
        # add skin and chest color to body
        skin_mat = colorize(body_obj)
        colorize_chest(body_obj, chest_mat)   

        # add body shape
        bm,mesh = get_bm()
        maxr = shape_body(bm)*rad
        bm.to_mesh(mesh)
        bm.free()
        
        x,y = sample_position(iceberg_bounds, positions, maxr)
        positions.append((x,y,maxr))
        body_obj.location = (x,y,cyl_scale)

        # add wings
        wing_obj = add_wing(offset=-maxr)
        colorize(wing_obj,skin_mat)
        wing_obj2 = add_wing(offset=+maxr)
        colorize(wing_obj2,skin_mat)

        # initialize head
        bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,2*cyl_scale+1),radius=rad)
        head_obj = bpy.context.active_object#data.objects[head_name]
        bm,mesh = get_bm()
        
        beak_faces = add_beak(bm)
        
        # head color
        colorize(head_obj,skin_mat)
        colorize_beak(head_obj,bm,beak_mat,beak_faces)

        bm.to_mesh(mesh)
        bm.free()

        # initialize eyes
        min_eye_rad = rad/6
        max_eye_rad = rad/3
        eye_rad = random.random()*(max_eye_rad-min_eye_rad)+min_eye_rad
        eye_pos1, eye_pos2 = eye_position(*head_obj.location,rad,0,1,eye_rad)
        left_eye = add_eye(eye_pos1,head_obj.location,eye_rad)
        colorize(left_eye,eye_mat)
        right_eye = add_eye(eye_pos2,head_obj.location,eye_rad)
        colorize(right_eye,eye_mat)


    bpy.ops.wm.save_as_mainfile(filepath='penguin.blend')
