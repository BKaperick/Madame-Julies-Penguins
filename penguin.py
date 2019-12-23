import bpy
import bmesh
import random
import sys
from math import sqrt,pi,sin,cos

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
    return radius_toxy(avg,r)

def radius_toxy(avg,r):
    angle = random.uniform(0,2*pi)
    return avg[0]+r*cos(angle),avg[1]+r*sin(angle)

def conify(mesh):
    bm = bmesh.new()



if __name__ == '__main__':
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
        cyl_scale = 2*random.random()
        x,y = sample_position(avg=(avgx,avgy))
        avgx = (avgx*cyl_count + x)/(cyl_count+1)
        avgy = (avgy*cyl_count + y)/(cyl_count+1)
        bpy.ops.mesh.primitive_cylinder_add(location=(x,y,0),radius=random.normalvariate(1,.1) + cyl_scale*.5 - .5)

        if cyl_count == 0:
            cyl_str = "Cylinder"
        else:
            cyl_str = "Cylinder." + str(cyl_count).zfill(3)
        bpy.data.objects[cyl_str].scale = (1,1,cyl_scale)
        bpy.data.objects[cyl_str].location = (x,y,cyl_scale)
        bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,2*cyl_scale+1))
        cyl_count+=1
    bpy.ops.wm.save_as_mainfile(filepath='penguin.blend')
