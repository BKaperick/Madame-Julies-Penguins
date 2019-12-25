import bpy
import bmesh
import random
import sys
from math import sqrt,pi,sin,cos
from matplotlib import pyplot as plt
import numpy as np
#from julia import Main#import julia
import mathutils
from functools import reduce

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

def sample_position(avg=(0,0),mean = 10,minr=5):
    k = 3*mean/2
    th = mean/k
    r = random.gammavariate(k,th) + minr
    return 0,0#radius_toxy(avg,r)

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
#        for i_e in e_indices:
#            bm.edges.ensure_lookup_table()
#            ed = f.edges[i_e]
#            edges.append(ed)
        for e in edges:
            e_direction = e.verts[1].co - e.verts[0].co
            # ignore top edges
            if e_direction.dot([0,0,1]) == 0 or not (e_direction[0] == 0 and e_direction[1] == 0):
                #print("rejected ", e.verts[0].co,e.verts[1].co,e_direction)
                continue
            maxv = e.verts[1]
            if e_direction[-1] < 0:
                maxv = e.verts[0]

            yield e,maxv

def extrude_cylinder_side(bm,r,h,smoothness=4,iters=100):
    #polycoeffs = get_side_curve(1,h)
    #knots,fvals = Main.poly_get_knots(polycoeffs)
    knots = [.5,.75]
    fvals = [1.2*r,1.1*r]

    knots = [.5]
    fvals = [1.2*r]
    vertice_lvls = dict()
    center_xy = mathutils.Vector((0,0))
    for e,v in vertical_edges(bm):
        print(e.verts[0].co)
        print(e.verts[1].co)
        print(v.co)
        for knot in knots:
            #enew,vnew = bmesh.utils.edge_split(e,max(e.verts,key=lambda v: v.co[0]),knot)
            e,vnew = bmesh.utils.edge_split(e,v,knot)
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
            if k_i == 1:
                for vvv in f.verts:
                    print(vvv.co)
                print("\n")
                print(vertices[curr].co)
                print(vertices[nextcurr].co)
                print("\n")
            #bmesh.utils.face_split(f,vertices[curr],vertices[nextcurr])

    for knot,fval in zip(knots,fvals):
        for v in vertice_lvls[knot]:
##            print("before:",v.co)
##            print(v.co.xy.length,fval*v.co.xy)
            v.co[0:2] = fval*(v.co.xy - center_xy)
#            print("after:",v.co)


def get_side_curve(r,h):
    a1 = random.random()*h
    a2 = random.random()*h
    a1p = min(a1,a2)
    a2 = max(a1,a2)
    a1 = a1p

    b1 = (1+random.random())*r
    b2 = random.random()*r
    b3 = (b2+.5*random.random())*r
    xr = [0,a1,a2,h]
    yr = [r,b1,b2,b3]
    p = np.polyfit(xr,yr,4)

    xx = np.linspace(0,h,50)
    #polynomial = lambda xx : p[0]*(xx**4) + p[1]*(xx**3) + p[2]*(xx**2) + p[3]*(xx) + p[4]*np.ones(xx.shape)
    #print(polynomial(xx))
    #plt.plot(xx,yy)
    #plt.show()
    return list(p)

    



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


if __name__ == '__main__':
    
    #Main.include('../Julia/interp.jl')
        
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
        cyl_scale = 2#2*random.random() + 1
        x,y = sample_position(avg=(avgx,avgy))
        avgx = (avgx*cyl_count + x)/(cyl_count+1)
        avgy = (avgy*cyl_count + y)/(cyl_count+1)
        rad = 2#random.normalvariate(1,.1) + cyl_scale*.5 - .5
        bpy.ops.mesh.primitive_cylinder_add(location=(x,y,0),radius=rad)

        if cyl_count == 0:
            cyl_str = "Cylinder"
        else:
            cyl_str = "Cylinder." + str(cyl_count).zfill(3)
        bpy.data.objects[cyl_str].scale = (1,1,cyl_scale)
        bpy.data.objects[cyl_str].location = (x,y,cyl_scale)
        
        mesh = bpy.context.object.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        #conify(bm,ratio=random.random())
        extrude_cylinder_side(bm,rad,cyl_scale)
        bm.to_mesh(mesh)
        bm.free()

        bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,2*cyl_scale+1))
        cyl_count+=1
    bpy.ops.wm.save_as_mainfile(filepath='penguin.blend')
