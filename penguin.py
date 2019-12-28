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


log_file = open('log.txt','w+')
def printl(*args):
    line = ' '.join([str(a) for a in args])
    log_file.write(line+'\n')
    print(line)

def angle_offset(vec1,vec2):
    cosang = vec1.dot(vec2)/(vec1.length*sqrt(sum([x**2 for x in vec2])))
    return acos(cosang)

def vector_sum(vecs):
    '''
    Built-in sum() doesnt handle objects which havent implemented __radd__ since it starts by computing `0+vecs[0]`
    '''
    return reduce((lambda x,y:x+y),vecs)

def sample_position_norm(m = 15, s=50):
    # with just one person, mean distance = m, zero support in collision.  Assume first at origin.  Assume symmetric for starters
    # E[x^2 + y^2] = m
    # 1. choose x at random, between [-2m,2m]
    # now, E[y^2] = m - x^2
    # 2. sample z~N(m-x^2,???)
    # 3. set y = sqrt(z)
    # 4. sample binary b~{-1,1} and set y = b*y
#    x = random.random()*2*m - m
#    z = random.normalvariate(m-x**2,10)
#    y = random.choice([-1,1])*(abs(z)**.5)
#    return (x,y)

    # 1. let r2~N(sqrt(m),s)^2 
    # 2. let x~unif[-r2,r2]
    # 3. let b~unif{-1,1}
    # 4. let y = b*(r2-x^2)
    r = random.normalvariate(m**.5, s)**2
    return radius_toxy(r)

def sample_position(avg=(0,0),mean = 20,minr=10):
    k = 3*mean/2
    th = mean/k
    r = random.gammavariate(k,th) + minr
    return radius_toxy(avg,r)

def radius_toxy(avg,r):
    angle = random.uniform(0,2*pi)
    return avg[0]+r*cos(angle),avg[1]+r*sin(angle)

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

def shape_body(bm,smoothness=24,iters=100):
    ycoeffs = get_side_curve()
    #knots,fvals = Main.poly_get_knots(ycoeffs,smoothness,iters)
    knots = [0,.2,.4,.7,1]
    fvals = [.9,1.1,1.5,1]
    knots = knots[1:-1]
    fvals = fvals[1:-1]
    knots = [1-k for k in knots[::-1]]
    fvals = fvals[::-1]
    return extrude_cylinder_side(bm,knots,fvals)

def colorize(color_scaled=None,mat=None):
    activeObject = bpy.context.active_object #Set active object to variable
    activeObject.data.materials.append(mat) #add the material to the object
    #mat.diffuse_color = (1, 5, .2, 1) #yellow
    #mat.diffuse_color = (1, .2, 5, 1) #violet
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
    if not color_scaled:
        color = random.choices(colors, probs)[0]
        color = [c + random.uniform(-10,10) for c in color]
        color_scaled = tuple([c/255 for c in color] + [1])

    mat.diffuse_color = color_scaled #blue
    return color_scaled

def colorize_chest(obj, epsilon = pi/6):
    matchest = bpy.data.materials.new(name="chest")
    matchest.diffuse_color = (1,1,1,1)
    obj.data.materials.append(matchest)
    count = [0,0]
    for i,f in enumerate(obj.data.polygons):
        if abs(angle_offset(f.normal,[0,1,0])) < epsilon:
            obj.data.polygons[i].material_index = 1
        else:
            obj.data.polygons[i].material_index = 0

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
            bmesh.utils.face_split(f,vertices[curr],vertices[nextcurr])

    for knot,fval in zip(knots,fvals):
        for v in vertice_lvls[knot]:
            v.co[0:2] = fval*v.co.xy
    return maxr

def add_beak(bm):
    v = max(bm.verts, key=lambda x: x.co[1])
    v.co[1] += 5


def get_side_curve(h=1):
    
    shoulder = 1.5 + random.random()
    waist = 1 + .4*random.random()
    bodylen = random.randint(8,15)
    body = [shoulder, shoulder+.2, shoulder+.3,shoulder+.35]
    body = body + [x for x in np.linspace(shoulder+.35,waist+.35,bodylen)]
    body = body + [waist+.3, waist+.2, waist]
    feet = []
    yr = body + feet
    #plt.plot(yr);plt.show()
    return yr

    



def conify(bm,ratio=.5):
    '''
    Transform upright cylinder into a conic-cylinder
    with top face `ratio` radius of bottom face
    '''
    for f in bm.faces:
        if f.normal.dot([0,0,1]) <= 0:
            continue
        center = vector_sum([v.co for v in f.verts])/len(f.verts)
        for v in f.verts:
            v.co[0:2] = ratio*(v.co.xy+center.xy)
        
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

if __name__ == '__main__':
    
    Main.include('interp.jl')
        
    # get rid of default cube at origin
    bpy.data.objects.remove(object=bpy.data.objects["Cube"])
    bpy.data.objects["Camera"].location[2] += 10
    bpy.data.objects["Light"].location = (0,0,15)
    bpy.data.objects["Light"].scale = (15,15,15)

    # number of grid points in x,y directions
    N = int(sys.argv[1])

    cyl_count = 0
    avgx = 0
    avgy = 0
    for i in range(N):
        cyl_scale = 2*random.random() + 6
        x,y = sample_position(avg=(avgx,avgy))
        avgx = (avgx*cyl_count + x)/(cyl_count+1)
        avgy = (avgy*cyl_count + y)/(cyl_count+1)
        rad = random.normalvariate(1,.1) + cyl_scale*.5 - .5
        bpy.ops.mesh.primitive_cylinder_add(location=(x,y,0),radius=rad)

        if cyl_count == 0:
            cyl_str = "Cylinder"
        else:
            cyl_str = "Cylinder." + str(cyl_count).zfill(3)
        bpy.data.objects[cyl_str].scale = (1,1,cyl_scale)
        bpy.data.objects[cyl_str].location = (x,y,cyl_scale)
        


        # edit body
        bm,mesh = get_bm()
        maxr = shape_body(bm)
        #colorize_chest(bm,mesh,mat)   
        

        bm.to_mesh(mesh)
        bm.free()
        mat = bpy.data.materials.new(name="skin_"+cyl_str) #set new material to variable
        color = colorize(mat=mat)
        
        obj = bpy.data.objects[cyl_str]
        #obj.data.materials.append(mat)


        colorize_chest(obj)
        add_wing(offset=-maxr*rad)
        colorize(color,mat)
        add_wing(offset=+maxr*rad)
        colorize(color,mat)

        obj = bpy.data.objects[cyl_str]

        # edit head
        bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,2*cyl_scale+1),radius=rad)
        bm,mesh = get_bm()
        add_beak(bm)
        bm.to_mesh(mesh)
        bm.free()
        colorize(color,mat)
        cyl_count+=1
    bpy.ops.wm.save_as_mainfile(filepath='penguin.blend')
