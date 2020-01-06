import bpy
import bmesh
import random
import sys
from math import sqrt,pi,sin,cos,acos
from matplotlib import pyplot as plt
import numpy as np
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

#def sample_position(avg=(0,0),mean = 20,minr=10):
#    k = 3*mean/2
#    th = mean/k
#    r = random.gammavariate(k,th) + minr
#    return 0,0#radius_toxy(avg,r)

#def radius_toxy(avg,r):
#    angle = random.uniform(0,2*pi)
#    return avg[0]+r*cos(angle),avg[1]+r*sin(angle)

def init_material(color, name):
    new_mat = bpy.data.materials.new(name=name)
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

def test_new_point(xy_bounds, circles, new_x, new_y, new_rad):
    for px,py,r in circles:
        d = sqrt((px - new_x)**2 +(py - new_y)**2)
        if d <= r+new_rad:
            return False
    return True

def sample_position(xy_bounds, circles, new_rad):
    is_valid = False
    while (not is_valid):
        x = random.random()
        y = random.random()
        linscale = lambda z,b: b[0] + z*(b[1]-b[0])
        x = linscale(x,xy_bounds[0])
        y = linscale(y,xy_bounds[1])
        is_valid = test_new_point(xy_bounds,circles,x,y,new_rad)
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
            #printl("returning edge direction {0} with verts: \n{1} and \n{2}".format(e_direction, maxv.co.xy,e.other_vert(maxv).co))
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
    probs = [100,10,100,10,10,10,10,10,10,100]
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
    #print(v.link_faces)
    #for f in v.link_faces:
    #    print(f)
    return v.index
    #return v.link_faces

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

if __name__ == '__main__':
    
    Main.include('interp.jl')
        
    # get rid of default cube at origin
    #bpy.data.objects.remove(object=bpy.data.objects["Cube"])
    bpy.data.objects["Camera"].location = (0,60,15)
    bpy.data.objects["Camera"].rotation_euler = (3*pi/2,pi,0)
    bpy.data.objects["Light"].location = (0,0,15)
    bpy.data.objects["Light"].scale = (15,15,15)

    # number of grid points in x,y directions
    N = int(sys.argv[1])

    cyl_count = 0
    avgx = 0
    avgy = 0
    
    chest_mat = init_material([1,1,1],"chest")
    beak_mat = init_material([1,140/255,0],"beak")
    ice_mat = init_material([1,1,1],"ice")
    sky_mat = init_material([101/255,216/255,1],"sky")
    
    # initialize iceberg
    iceberg_obj = bpy.data.objects['Cube']
    iceberg_obj.scale = (50,50,5)
    iceberg_obj.location = (0,0,-iceberg_obj.scale[2])
    colorize(iceberg_obj, ice_mat)
    iceberg_bounds = ((-50,50),(-50,50))

    # initialize sky
    sz = 150
    bpy.ops.mesh.primitive_plane_add(size=sz, location=(0,-20,sz/2-50),rotation=(pi/2,0,0))
    sky_obj = bpy.data.objects['Plane']
    colorize(sky_obj, sky_mat)
 
    positions = []
    
    for i in range(N):
        cyl_scale = 2*random.random() + 6
        rad = random.normalvariate(1,.1) + cyl_scale*.5 - .5
        bpy.ops.mesh.primitive_cylinder_add(location=(0,0,0),radius=rad)

        if cyl_count == 0:
            body_name = "Cylinder"
            head_name = "Sphere.002"
        else:
            body_name = "Cylinder." + str(cyl_count).zfill(3)
            head_name = "Sphere." + str(3*cyl_count+2).zfill(3)
        
        body_obj = bpy.data.objects[body_name]
        body_obj.scale = (1,1,cyl_scale)
        
        # add skin and chest color to body
        #skin_mat = bpy.data.materials.new(name="skin_"+body_name) #set new material to variable
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
        head_obj = bpy.data.objects[head_name]
        bm,mesh = get_bm()
        
        beak_faces = add_beak(bm)
        
        # head color
        colorize(head_obj,skin_mat)
        colorize_beak(head_obj,bm,beak_mat,beak_faces)

        bm.to_mesh(mesh)
        bm.free()
        cyl_count+=1

    bpy.ops.wm.save_as_mainfile(filepath='penguin.blend')
