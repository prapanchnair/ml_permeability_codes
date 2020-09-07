import numpy as np
import matplotlib
#print ("Matplotlib Version :",matplotlib.__version__)
import pylab as pl
import time, sys, os
import decimal
import glob

from subprocess import call
from IPython.display import Image
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
#from sympy import *
#from mpmath import quad
from scipy.integrate import quad
import random
import string

vol_frac = 0.5
radius_cyl = np.sqrt(vol_frac/np.pi)
rho = 1000
mu = 0.001
L = 2*radius_cyl 

def Reynolds( V_mean, L, rho=1000, mu=0.001):
    Re_actual = rho*V_mean*L/mu
    return Re_actual

def majorAxis(alpha):
    return np.sqrt((0.5/np.pi)/alpha)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)
        
def plot_fourier_curve(shape):
    # input( coeffs ): the fourier coefficients of dimension 2,2*M+1, where M is the maximum degree.
    # output plot: Plots the shape 
    coeffs = shape["coeffs"]
    name =shape["name"]
    x_coeffs = coeffs[0,:]
    y_coeffs = coeffs[1,:]
    M = (np.shape(coeffs)[1] -1 ) // 2
    start_t = 0.0
    t = np.linspace(start_t,start_t+2.0*np.pi,num=100,endpoint=True)
    #print((t))
    x = np.zeros(np.shape(t))
    y = np.zeros(np.shape(t))
    x += 0.5*x_coeffs[0] ; y += 0.5*y_coeffs[0]
    for mi in range(1,M+1):
        x += x_coeffs[2*mi-1]*np.cos(mi*t) + x_coeffs[2*mi]*np.sin(mi*t)
        y += y_coeffs[2*mi-1]*np.cos(mi*t) + y_coeffs[2*mi]*np.sin(mi*t)
  
    pl.plot(x,y,'k-')
    head = "shape "+name
    curve = np.column_stack((x,y))
    np.savetxt(name,curve,delimiter=" ")#,header=head)
    pl.axis('equal')
    pl.title('Shape from Fourier Coeffs.')
    pl.show()
    coords = {"x":x,
              "y":y}
    return coords

def minkowski_fourier_curve(coeffs):
    # input( shape ): contains the key "coeffs" -the fourier coefficients of dimension 2,2*M+1, where M is the maximum degree.
    #                 and the key "name" for shape name.
    # output (W) : Dictionary containing the four 2D minkowski tensors W020, W120, W220, W102 and the area 
    # and perimeter of the curve/shape.
    #coeffs = shape["coeffs"]
    t=symbols("t") # parameter of the curve
    x_coeffs = coeffs[0,:]
    y_coeffs = coeffs[1,:]
    # m =0 , zeroth degree terms, also gives the centroid of the shape.
    expr_X = "0.5*"+str(coeffs[0,0])
    expr_Y = "0.5*"+str(coeffs[1,0])
    M = (np.shape(coeffs)[1] -1)//2
    # X and Y coodinates as parametric representation using fourier series.
    for mi in range(1,M+1):
        expr_X += "+" + str(x_coeffs[2*mi-1]) + "*cos("+str(mi)+"*t) + " +str(x_coeffs[2*mi])+"*sin("+str(mi)+"*t)" 
        expr_Y += "+" + str(y_coeffs[2*mi-1]) + "*cos("+str(mi)+"*t) + " +str(y_coeffs[2*mi])+"*sin("+str(mi)+"*t)" 
    # derivative terms required for normal and curvature computation
    sym_x = sympify(expr_X)
    sym_y = sympify(expr_Y)
    # dx/dt
    sym_dx = diff(sym_x,t)
    # d^2x/dt^2
    sym_ddx = diff(sym_dx,t)
    # dA = ydx infinitesimal area
    sym_ydx = sym_y*sym_dx
    
    sym_dy = diff(sym_y,t)
    sym_ddy = diff(sym_dy,t)
    # ds = sqrt(x'^2 + y'^2) , the infinitesimal arc-length
    sym_ds = sqrt(sym_dx**2 + sym_dy**2)
    # position vector r
    sym_r = [sym_x, sym_y]
    # unit normal vector n
    sym_norm_mag = sqrt(sym_dx**2 + sym_dy**2)
    sym_norm = [sym_dx/sym_norm_mag, sym_dy/sym_norm_mag]
    #print("Computed derivatives")
    # Area = \int ydx
    area = Integral(sym_ydx,(t,0,2*pi)).evalf(5)
    perimeter = Integral(sym_ds,(t,0,2*pi)).evalf(5)
    kappa = (sym_dx*sym_ddy - sym_dy*sym_ddx)/(sym_dx**2 + sym_dy**2)**(3/2)
    #print("Computing integrals ...")
    #Initialize the minkowski tensors 
    W020 = np.zeros((2,2))
    W120 = np.zeros((2,2))
    W220 = np.zeros((2,2))
    W102 = np.zeros((2,2))
    x = symbols('x')
    #tensor computation
    for ia in range(2):
        for ib in range(2):
#             W020[ia,ib] = Integral(sym_r[ia]*sym_r[ib]*sym_ydx, (t,0,2*pi)).evalf(5)
#             print("Computing W120 ...")
#             W120[ia,ib] = 0.5* Integral(sym_r[ia]*sym_r[ib]*sym_ds, (t,0,2*pi)).evalf(5)
#             W220[ia,ib] = 0.5* Integral(kappa*sym_r[ia]*sym_r[ib]*sym_ds, (t,0,2*pi)).evalf(5)
#             print("Computing W102 ...")
#             W102[ia,ib] = 0.5* Integral(sym_norm[ia] * sym_norm[ib]*sym_ds,(t,0,2*pi)).evalf(5)
            f = lambdify(t,sym_r[ia]*sym_r[ib]*sym_ydx)
            W020[ia,ib],err = quad( f, 0,2*np.pi) 
            #print(W020[ia,ib])
            #print("Computing W120 ...")
            f = lambdify(t,sym_r[ia]*sym_r[ib]*sym_ds)
            W120[ia,ib],err = quad(f, 0,2*np.pi)
            W120[ia,ib] = 0.5* W120[ia,ib]
            f = lambdify(t,kappa*sym_r[ia]*sym_r[ib]*sym_ds)
            W220[ia,ib],err = quad(f, 0,2*np.pi)
            W220[ia,ib] = 0.5*W220[ia,ib]
            #print("Computing W102 ...")
            f = lambdify(t,sym_norm[ia] * sym_norm[ib]*sym_ds)
            W102[ia,ib], err =  quad(f, 0,2*np.pi)
            W102[ia,ib] = 0.5* W102[ia,ib]
    #dictionary with computed quantities        
    W={"W020":W020,
       "W120":W120,
       "W220":W220,
       "W102":W102,
       "area":area,
       "perimeter":perimeter
       }
    
    return W 

# def simulate_flow():
#     DIR = './shapes/coords'
#     createFolder('./simulations')
#     start_t = time.time()
#     name_list = []
#     num = 0
#     #n_angles =20
#     n_shapes = len(os.listdir(DIR))
#     for name in os.listdir(DIR):
#         if os.path.isfile(os.path.join(DIR,name)):
#             update_progress(num/n_shapes,start_t,time.time())
#             num += 1
#             thisfolder ='./simulations/'+name
#             createFolder(thisfolder)
#             #print("Shape No. "+str(num)+" : "+name)
#             #for angle in range(n_angles):
#             #theta = random.uniform(0.0,np.pi)
#             #    thisfolder ='./simulations/'+name+'/theta_'+str(round(theta,3))
#             #    createFolder(thisfolder)
#             call(["cp","vorticity.gfs",thisfolder+'/.'])
#             call(["cp","xprofile",thisfolder+'/.'])
#             f=open(thisfolder+"/shape.gts","w")
#             call(["shapes",os.path.join(DIR,name)],stdout=f) #+" > "+thisfolder+"/shape.gts"])
#             os.chdir(thisfolder)
#             call(["gerris2D","vorticity.gfs"])
#             #xp = (np.loadtxt('xprof', delimiter=" "))
             
#             #pl.plot(xp[:,6],xp[:,2],label=r'$\theta =$') #thets
#                 #Vel_mean[i,1] = np.mean(xp[:,6])
#                 #Vel_mean[i,0] = theta
#             #Image("velocity.png")
#             os.chdir('../../')
               
#             #name_list.append(name)
    
#     n_simulations = n_shapes

def simulate_flow(dp=0.000001,DIR='./shapes_low0/coords'):
#    DIR = './shapes_low0/coords'
#    dp_0 = 0.000001
#    p_ratio = round(dp_0/dp,2)
    dp_string = '{:.0e}'.format(decimal.Decimal(str(dp)))
    folder_name ='./simulations_dP_'+dp_string
    input_file  ='vorticity_'+dp_string+'.gfs'
    with open('vorticity.gfs','r') as fin:
#     # with is like your try .. finally block in this case
        input_string = fin.readlines()
    for index, line in enumerate(input_string):
         if line.strip().startswith('Source {} U'):
                input_string[index] = 'Source {} U '+str(dp)
    with open(input_file, 'w') as file:
         file.writelines( input_string )
    
    
    createFolder(folder_name)
    start_t = time.time()
    name_list = []
    num = 0
    #n_angles =20
    n_shapes = len(os.listdir(DIR))
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR,name)):
            update_progress(num/n_shapes,start_t,time.time())
            num += 1
            thisfolder =folder_name + '/' + name
            createFolder(thisfolder)
            #print("Shape No. "+str(num)+" : "+name)
            #for angle in range(n_angles):
            #theta = random.uniform(0.0,np.pi)
            #    thisfolder ='./simulations/'+name+'/theta_'+str(round(theta,3))
            #    createFolder(thisfolder)
            call(["cp", input_file ,thisfolder+'/.'])
            call(["cp","xprofile",thisfolder+'/.'])
            f=open(thisfolder+"/shape.gts","w")
            call(["shapes",os.path.join(DIR,name)],stdout=f) #+" > "+thisfolder+"/shape.gts"])
            os.chdir(thisfolder)
            call(["gerris2D",input_file])
            #xp = (np.loadtxt('xprof', delimiter=" "))
             
            #pl.plot(xp[:,6],xp[:,2],label=r'$\theta =$') #thets
                #Vel_mean[i,1] = np.mean(xp[:,6])
                #Vel_mean[i,0] = theta
            #Image("velocity.png")
            os.chdir('../../')
               
            #name_list.append(name)
    
    n_simulations = n_shapes
    
def fourier2Cart(coeffs,t):
    #x_coeffs = coeffs[0,:]
    #y_coeffs = coeffs[1,:]
    #M = (np.shape(coeffs)[1] -1 ) // 2
    #x = np.zeros(np.shape(t))
    #y = np.zeros(np.shape(t))
    #x += 0.5*x_coeffs[0] ; y += 0.5*y_coeffs[0]
    #for mi in range(1,M+1):
    #    x += x_coeffs[2*mi-1]*np.cos(mi*t) + x_coeffs[2*mi]*np.sin(mi*t)
    #    y += y_coeffs[2*mi-1]*np.cos(mi*t) + y_coeffs[2*mi]*np.sin(mi*t)
    #t.reshape(len(t))
    #t=t[:,np.newaxis].T
    tt = np.row_stack((t,t))
    #print(np.shape(tt))
    coords = np.zeros(np.shape(tt))
    coords += 0.5*coeffs[:,0,np.newaxis]
    M = (np.shape(coeffs)[1] -1 ) // 2
    for mi in range(1,M+1):
        coords += coeffs[:,2*mi-1,np.newaxis]*np.cos( mi*tt) + coeffs[:,2*mi,np.newaxis]*np.sin(mi*tt)
    #coords = np.row_stack((x,y))    
    return coords

def generateShape(res=200,M=4):
    t = np.linspace(0, 2.0*np.pi, num=res, endpoint=True)    
    dt = t[1]-t[0]
    coeffs = np.zeros((2,2*M+1))
    bad_shape = True
    n_attempts = 0
    while bad_shape == True:
        alpha = np.random.uniform(1.0,2.0)
        a = majorAxis(alpha)
        b = alpha*a
        #a = 1
        #b = 1
        #print("the major and minor axes are:"+str(a)+","+str(b))
        coeffs[0,1] = a # create an ellipse as starting point
        coeffs[1,2] = b # create an ellipse as starting point
        coeffs[:,3::] = coeffs[:,3::] + 0.25*a*(np.random.rand(2,2*M-2) -0.5)#-0.5    
        coords = fourier2Cart(coeffs,t)
        #pl.plot(coords[0,:],coords[1,:],'-')
        
        dx  = np.gradient(coords,axis=1)
        ddx = np.gradient(dx, axis=1)
       
        num   = dx[0,:] * ddx[1,:] - ddx[0,:] * dx[1,:]
        denom = dx[0,:] * dx[0,:]  + dx[1,:] * dx[1,:]
        denom = np.sqrt(denom)
        denom = denom * denom * denom
        curvature = num / denom

        sharp_edge = False
        outside_domain = False
        if (np.amax(np.absolute(curvature)) > 20):
            sharp_edge = True
        coords_prime = np.gradient(coords,dt,axis=1)
        integrand = coords_prime[1,:] * coords[0,:]
        area = np.trapz(integrand, x=t)
        scale = np.sqrt(0.5 / np.absolute(area))
        coeffs = scale * coeffs
        coords = fourier2Cart(coeffs,t)
        if(np.any(np.abs(coords) >= 0.5)):
            outside_domain = True
        bad_shape = sharp_edge or outside_domain
        n_attempts +=1
        #if(bad_shape):
        #   print( "This shape is bad:"+str(sharp_edge)+str(outside_domain))
           
    #x_coeffs_prime = x_coeffs[1:]
    #y_coeffs_prime = y_coeffs[1:]

   
    coords_prime = np.gradient(coords,dt,axis=1)

    integrand = coords[1,:] * coords_prime[0,:]
    area = np.trapz(integrand, x=t)

#    x = np.append(x, x[0])
#    y = np.append(y, y[0])

    length  = np.sum( np.sqrt(np.ediff1d(coords[0,:]) * np.ediff1d(coords[0,:]) + np.ediff1d(coords[1,:]) * np.ediff1d(coords[1,:])) )

    print('x-coefficients: ' + str(coeffs[0,:]))
    print('y-coefficients: ' + str(coeffs[1,:]))
    print('enclosed area:  '   + str(np.absolute(area)))
    print('curve length:   '   + str(length))
    shape={"coeffs":coeffs,
           "coords":coords}
    pl.plot(coords[0,:],coords[1,:],'-')
    return shape

def check_self_intersection(coords):
    result = False
    for i in range(2,np.shape(coords)[1]-1):
        
        p = coords[:,i]
        dp = coords[:,i+1] - p 
        for j in range(0,i-2):
            if (result==False):
                q = coords[:,j]
                dq = coords[:,j+1] - q
                dpdq = np.cross(dp,dq)
                t = np.cross(q-p,dq)/dpdq
                u = np.cross(q-p,dp)/dpdq
                if(dpdq != 0):
                    if(0<= t <= 1):
                        if(0<= u <= 1):
                            result = True
                            
    return result
        
        
def check_domain_intersection(coords):
    result = np.any(np.abs(coords)>= 0.5)
    return result
def generate_Npoint_shape(N=10,M=4, res=100):
    #random.seed(1516)
    bad_shape =True
    while bad_shape == True:
        pos_r = np.random.uniform(0,0.5,(N))
        #pos_thet = np.random.uniform(0,2*np.pi,(1,N))
        pos_thet =np.linspace(0,2*np.pi,num=N,endpoint=False)
        posx = pos_r*np.cos(pos_thet)
        posy = pos_r*np.sin(pos_thet)
        pos = np.row_stack((posx,posy))
        center = np.mean(pos,axis=1)
        r = pos - center[:,np.newaxis]
        r_mag  = np.sqrt(r[0,:]**2 + r[1,:]**2)
        x = np.zeros((2,np.shape(r)[1]))
        x[0,:] = 1
        costh =np.diag(np.matmul(r.T,x))#r.x
        costh = costh/r_mag
        theta = np.arccos(costh)
        ry = r[1,:]
        rx = r[0,:]
        neg = np.where(ry<0)
        theta[neg] = 2*np.pi - theta[neg]
        #print(r)
        #print(theta)
        rx = rx[np.argsort(theta)]
        ry = ry[np.argsort(theta)]
        theta = theta[np.argsort(theta)]
        #print(rx,ry)
        #print(theta)
        b = np.append(rx,ry)
        #print(np.shape(b))
        #M = 4
        m =  2*M+1
        A = np.zeros((N,m))
        A[:,0] = 1.0
        for j in range(1,M+1):
            A[:,2*j-1] = np.cos(j*theta)
            A[:,2*j]   = np.sin(j*theta)
        # Use the same A for both x and y coordinates.

        AA = np.matmul(A.T,A)
        #print("solving")
        #print(np.shape(AA))
        #print(np.shape(rx))
        coeffs_x = np.linalg.solve(AA,np.matmul(A.T,rx))
        coeffs_y = np.linalg.solve(AA,np.matmul(A.T,ry))
        coeffs = np.row_stack((coeffs_x,coeffs_y))
        #oeffs = scale_area(coeffs)
            #coeffs[:,2*mi-1,np.newaxis]*np.cos( mi*tt) + coeffs[:,2*mi,np.newaxis]*np.sin(mi*tt)
        #np.cos(M*theta)
        t = np.linspace(0, 2.0*np.pi, num=res, endpoint=True)
        dt = t[1]-t[0]
        coords = fourier2Cart(coeffs,t)
        coords_prime = np.gradient(coords,dt,axis=1)
        integrand = coords_prime[1,:] * coords[0,:]
        area = np.trapz(integrand, x=t)
        self_intersection = check_self_intersection(coords) 
        scale = np.sqrt(0.5 / np.absolute(area))
        coeffs = scale * coeffs
        coords = fourier2Cart(coeffs,t)
        domain_intersection = check_domain_intersection(coords)
        #bad_shape = False
        bad_shape = self_intersection or domain_intersection
        
#    pl.figure(figsize=(8,4))
#    pl.subplot(121,projection='polar')
#    pl.plot(pos_thet,pos_r,'o')
#    pl.grid(True)
#    pl.subplot(122)
#    pl.axis('equal')
#     pl.xlim(-0.5,0.5)
#     pl.ylim(-0.5,0.5)
#     pl.plot(r[0,:],r[1,:],'o')
#     pl.plot(coords[0,:],coords[1,:],'r-')
       
    shape={"coeffs":coeffs,
           "coords":coords}
    
    #pl.plot(coords[0,:],coords[1,:],'-')
    return shape
def plot_shapes(shapes):
    nr = len(shapes)//5
    width = 10
    height = nr*width/5
    pl.figure(figsize=(width,height))
    for i in range(len(shapes)):
        
        showImageinArray(i,len(shapes),shapes[i]["coords"])
        
def showImageinArray(i,N,coords):
    #fig = figure(figsize=(6,6))
    #number_of_files = len(list_of_files)
    #print(number_of_files)
    im_per_row = 5
    numrows = N // im_per_row
    #remaining = i % im_per_row
    #for i in range(numrows+1):
    #    for j in range(im_per_row):      
    #        k = i*im_per_row + j
    #        if (k<number_of_files):
    pl.subplot(numrows+1,im_per_row,i+1)
    pl.plot(coords[:,0],coords[:,1],'r-')
    pl.plot(axis='equal')
    pl.subplots_adjust(bottom=0.0)
    
    pl.axis('off')
    #pl.show()
    
def update_progress(progress, start, now):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    if(progress == 0 ):
        time_left = -1
    else:
        time_left = (now-start)/progress - (now-start)
    text = "\rPercent: [{0}] {1}% {2} {3} min".format( "#"*block + "-"*(barLength-block), round(progress*100,2), status,round(time_left/60,2))
    sys.stdout.write(text)
    sys.stdout.flush()

def write_shape(shape):
    #N = len(shapes)
    #random.seed(1516)
    createFolder('./shapes')
    coeff_folder = "./shapes/"+"coeffs/"
    coord_folder = "./shapes/"+"coords/"
    createFolder(coeff_folder)
    createFolder(coord_folder)
    #for i in range(len(shapes)):
    name = id_generator()
    #shape["name"] = name
    coord_file = coord_folder+name
    coeff_file = coeff_folder+name
    np.savetxt(coord_file,shape["coords"].T,delimiter=' ')
    np.savetxt(coeff_file,shape["coeffs"].T,delimiter=' ')
    return name    
        
def id_generator(size=6, chars=string.ascii_uppercase+string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def read_shapes():
    DIR = './shapes/coeffs'
    #createFolder('./simulations')
    shapes=[]
  
    num = 0
    #n_angles =20
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR,name)):
            coeffs = np.loadtxt(name,delimiter=' ')
            coeffs = coeffs.T
            shape= {"coeffs": coeffs}
            shapes.append(shape)
    return shapes
def compute_minkowski_tensors():
    DIR = './shapes/coeffs'
    mt_folder = './shapes/MT'
    createFolder(mt_folder)
    start_t = time.time()
    name_list = []
    num = 0
    n_shapes = len(os.listdir(DIR))
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR,name)):
            update_progress(num/n_shapes,start_t, time.time())
            num += 1
            coeffs = np.loadtxt(os.path.join(DIR,name),delimiter=' ')
            coeffs = coeffs.T
            W = minkowski_fourier_curve(coeffs)    
            W_write = np.row_stack((W["W020"],W["W120"],W["W220"],W["W102"]))
            mt_file = os.path.join(mt_folder,name)
            np.savetxt(mt_file,W_write,delimiter=' ')

def consolidate_coords():
    DIR = './shapes/coords'
    createFolder('./shapes/coords_consolidated')
    n_shapes= len(os.listdir(DIR))
    allnames = os.listdir(DIR)
    allnames.sort()
    Xcoord =np.zeros((n_shapes,100)) 
    Ycoord =np.zeros((n_shapes,100))
    length =np.zeros((n_shapes,1))
    for counter,name in enumerate(allnames):
        localfile = os.path.join(DIR,name)
        arr = np.loadtxt(localfile)
        
        npoints = np.shape(arr)[0]
        # find total length
        len_ = 0.0
        for j in range(npoints-1):
            len_ =  len_ + np.sqrt((arr[j+1,0]-arr[j,0])**2 + (arr[j+1,1]-arr[j,1] )**2 )
        
        if (npoints == 200):
            arr_X=arr[::2,0]
            arr_Y=arr[::2,1]
        elif (npoints == 300):
            arr_X=arr[::3,0]
            arr_Y=arr[::3,1]
        else :
            arr_X = arr[:,0]
            arr_Y = arr[:,1]
        assert np.shape(arr_X)[0]==100 , " Array x problem"
        assert np.shape(arr_Y)[0]==100 , " Array y problem"
        Xcoord[counter,:] = arr_X
        Ycoord[counter,:] = arr_Y
        length[counter,0] = len_
    np.savetxt('./shapes/coords_consolidated/Xcoord',Xcoord,delimiter=' ')
    np.savetxt('./shapes/coords_consolidated/Ycoord',Ycoord,delimiter=' ')
    np.savetxt('./shapes/coords_consolidated/length',length,delimiter=' ')
    
            
            
def consolidate_papaya_minkowskis():
    DIR = './shapes_low0/papaya_out'
    createFolder('./shapes_low0/papaya_consolidated')
    n_shapes= len(os.listdir(DIR))
    allnames = os.listdir(DIR)
    allnames.sort()
    W102 =np.zeros((n_shapes,12)) 
    W020 =np.zeros((n_shapes,12))
    W120 =np.zeros((n_shapes,12))
    W220 =np.zeros((n_shapes,12))
    W211 =np.zeros((n_shapes,12))
    for counter,name in enumerate(allnames):
        localdir = os.path.join(DIR,name)
        arr_020 = np.loadtxt(os.path.join(localdir,'tensor_W020.out'))
        arr_120 = np.loadtxt(os.path.join(localdir,'tensor_W120.out'))
        arr_220 = np.loadtxt(os.path.join(localdir,'tensor_W220.out'))
        arr_102 = np.loadtxt(os.path.join(localdir,'tensor_W102.out'))
        arr_211 = np.loadtxt(os.path.join(localdir,'tensor_W211.out'))
        W020[counter,:] = arr_020
        W120[counter,:] = arr_120
        W220[counter,:] = arr_220
        W102[counter,:] = arr_102
        W211[counter,:] = arr_211
    np.savetxt('./shapes_low0/papaya_consolidated/W_020',W020,delimiter=' ')
    np.savetxt('./shapes_low0/papaya_consolidated/W_102',W102,delimiter=' ')
    np.savetxt('./shapes_low0/papaya_consolidated/W_120',W120,delimiter=' ')
    np.savetxt('./shapes_low0/papaya_consolidated/W_220',W220,delimiter=' ')
    np.savetxt('./shapes_low0/papaya_consolidated/W_211',W211,delimiter=' ')
            
def compute_eigens_of_minkowskis():
    DIR = './shapes/MT'
    createFolder('./shapes/eigens')
    n_shapes = len(os.listdir(DIR))
    allnames = os.listdir(DIR)
    allnames.sort()
    start_t = time.time()
    name_list = []
    num = 0
    W020 = np.zeros((2,2))
    W120 = np.zeros((2,2))
    W220 = np.zeros((2,2))
    W102 = np.zeros((2,2))
    W020_eig = np.zeros((n_shapes,3))
    W120_eig = np.zeros((n_shapes,3))
    W220_eig = np.zeros((n_shapes,3))
    W102_eig = np.zeros((n_shapes,3))
    vec_x = np.array([1.0, 0])
    #n_angles =20
    print("Total shapes" + str(n_shapes));
    for name in allnames:
        if os.path.isfile(os.path.join(DIR,name)):
            update_progress(num/n_shapes,start_t,time.time())            
            Ws = np.loadtxt(os.path.join(DIR,name),delimiter=' ')
            W020=-1*Ws[0:2,:]; 
            W120=Ws[2:4,:];
            W220=Ws[4:6,:];
            W102=Ws[6:8,:];
            
            Eval_020,Evec_020 = np.linalg.eigh(W020)
            Eval_120,Evec_120 = np.linalg.eigh(W120)
            Eval_220,Evec_220 = np.linalg.eigh(W220)
            Eval_102,Evec_102 = np.linalg.eigh(W102)
            
            thet_020 = np.dot(Evec_020[:,1], vec_x)
            thet_120 = np.dot(Evec_120[:,1], vec_x)
            thet_220 = np.dot(Evec_220[:,1], vec_x)
            thet_102 = np.dot(Evec_102[:,1], vec_x)
            W020_eig[num, :] = [Eval_020[0] , Eval_020[1], thet_020]
            W120_eig[num, :] = [Eval_120[0] , Eval_120[1], thet_120]
            W220_eig[num, :] = [Eval_220[0] , Eval_220[1], thet_220]
            W102_eig[num, :] = [Eval_102[0] , Eval_102[1], thet_102]
            num += 1
            
    np.savetxt(os.path.join('./shapes/eigens','W020'),W020_eig,delimiter=' ')
    np.savetxt(os.path.join('./shapes/eigens','W120'),W120_eig,delimiter=' ')
    np.savetxt(os.path.join('./shapes/eigens','W220'),W220_eig,delimiter=' ')
    np.savetxt(os.path.join('./shapes/eigens','W102'),W102_eig,delimiter=' ')          


# Compute the minkowski tranformation of along-stream and cross dream unit vector
def compute_transformations_minkowskis():
    DIR = './shapes/MT'    
    n_shapes = len(os.listdir(DIR))
    allnames = os.listdir(DIR)
    allnames.sort()
    start_t = time.time()
    name_list = []
    num = 0
    W020 = np.zeros((2,2))
    W120 = np.zeros((2,2))
    W220 = np.zeros((2,2))
    W102 = np.zeros((2,2))
    W = np.zeros((8,2))
    trans_x = np.zeros((n_shapes,8))
    trans_y = np.zeros((n_shapes,8))
    
    # along stream and across stream vector
    vec_x = np.array([1.0, 0])
    vec_y = np.array([0.0, 1.0])
    
    print("Total shapes" + str(n_shapes));
    for name in allnames:
        if os.path.isfile(os.path.join(DIR,name)):
            update_progress(num/n_shapes,start_t,time.time())            
            Ws = np.loadtxt(os.path.join(DIR,name),delimiter=' ')
            W[0:2,:] = -1*Ws[0:2,:]; 
            W[2:8]   = Ws[2:8,:];
            trans_x[num,:] =  (np.matmul(W, vec_x.T)).T
            trans_y[num,:] =  (np.matmul(W, vec_y.T)).T            
            num += 1
            
    np.savetxt('./MT_transform_x',trans_x, delimiter=' ')
    np.savetxt('./MT_transform_y',trans_y, delimiter=' ')
            
        
def showImagesHorizontally(list_of_files):
    fig = figure(figsize=(8,10))
    number_of_files = len(list_of_files)
    print(number_of_files)
    im_per_row = 5
    numrows = number_of_files // im_per_row
    remaining = number_of_files % im_per_row
    for i in range(numrows+1):
        for j in range(im_per_row):      
            k = i*im_per_row + j
            if (k<number_of_files):
                a=fig.add_subplot(numrows+1,im_per_row,k+1)
                fig.subplots_adjust(bottom=0.0)
                image = imread(list_of_files[k]+"/velocity.png")
                imshow(image)
                axis('off')
        
def average_velocities():
    
    DIR_ = glob.glob('./simulations_dP_*')
    dP_ = []    
    dpstr_ = []
    for k, directory in enumerate(DIR_):
        dpstr_.append(directory.rsplit("_",1)[1])
        dP_.append(float(dpstr_[k]))
    dP,DIR,dpstr = zip(*sorted(zip(dP_,DIR_,dpstr_)))

    #DIR.sort()
    print("directories considered:")
    print(DIR)
    num= 11110   
    V_mean = np.zeros((num,len(DIR)+1))
    
       
    header_text=" 0 "
    for k, directory in enumerate(DIR):
        folders= os.listdir(directory)
        num_folders = len(folders)
        header_text += dpstr[k] + " "
        if(num == num_folders):
            folders.sort()
            
            for index, folder in enumerate(folders):
            #dP = dP_0
               if( os.path.isfile(os.path.join(directory+'/'+folder,'xprof')) == False):
#                badfolder = folder
                   print(folder, index,k)
#                count += 1
                   V_mean[index,k+1] = -0.00001
               else :
                   flowdata = np.loadtxt(os.path.join(directory+'/'+folder,'xprof'),delimiter=' ')
                   V_mean[index,k+1] = np.mean(flowdata[:,6])
        else :
            V_mean[:,k+1] = -1
    
    shape_names = os.listdir('./shapes/coeffs')
    shape_names.sort()
    modes =np.zeros((num,1))
    for k, file in enumerate(shape_names):
        thisfile = os.path.join('./shapes/coeffs',file)
        modes[k,0] = int(file_len(thisfile))
        
    flow_data = np.column_stack((V_mean, modes))
    np.savetxt('flow_mean_velocity',flow_data,delimiter=' ', header=header_text + " modes" )
    
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1   
def compute_permeabilities(order=2,points=3):
    vol_frac = 0.5
    radius_cyl = np.sqrt(vol_frac/np.pi)
    rho = 1000
    mu = 0.001
    L =radius_cyl 
    vel_mean = np.loadtxt('flow_mean_velocity',delimiter=' ')
    x = vel_mean[:,0:points] # 11110 x 3
    num_shapes = np.shape(x)[0]
    DIR_ = glob.glob('./simulations_dP_*')
    #dP_ = np.zeros((1,int(len(DIR_))+1))
    dP_ = []
    dP_.append(float(0.0))
    dpstr_ = []
    dpstr_.append('0.0')
    for k, directory in enumerate(DIR_):
        dpstr_.append(directory.rsplit("_",1)[1])
        dP_.append((float(dpstr_[k+1])))
    dP= sorted(dP_)
    print(dP_)
    print(dP)
    y = np.array(dP[0:points])    # 3
    
    
    id_max = np.argmin(x[:,points-1])
    x_points = np.linspace(x[id_max,0],x[id_max,points-1],25)
    x_nd = rho*x*L/mu
    x_points_nd = rho*x_points*L/mu
    i=points-1
    XX = []
    p_all =[]
    print("Max Re for dp="+str(dP[i])+"is: ",x_nd[id_max,i], id_max)
    for k in range(num_shapes):
        XX.append(np.vstack((x[k,:]**2, x[k,:],np.zeros_like(x[k,:]))).T)
        p_all.append(np.linalg.lstsq(XX[k],y)[0])
    test_1 =150    
    print((p_all[test_1]))
    p_k = np.array(p_all)
    print(p_k[test_1,:])
    y_ls = np.polyval(p_all[test_1],x_points)    
    pl.plot(x_points[0:15], y_ls[0:15], 'k-',label='S1')
    pl.plot(x[test_1,0:points],y[0:points], 'k.',markersize=15)
    
    test_2 = 99990
    print((p_all[test_2]))
    p_k = np.array(p_all)
    print(p_k[test_2,:])
    y_ls = np.polyval(p_all[test_2],x_points)    
    pl.plot(x_points[0:15], y_ls[0:15], 'r-',label='S2')
    pl.plot(x[test_2,0:points],y[0:points], 'r.',markersize=15)
    
    pl.legend()
    pl.show()
    np.savetxt('fit_coeffs_order_'+str(order)+'_'+str(points),p_k)
    #print(np.shape(XX))

def plot_shapes(id):
    # id is the list of the shape numbers 
    vel_mean = np.loadtxt('flow_mean_velocity',delimiter=' ')
    fit_data = np.loadtxt('fit_coeffs_order_2_6',delimiter=' ')
    DIR_ = glob.glob('./simulations_dP_*')
    ref_DIR = glob.glob('./ref_shapes/simulations_dP_*')
    
    #####reference shape computations
    num_ref = len(ref_DIR)
    vel_ref_mean = np.zeros((num_ref+1, 1))
    vel_ref_mean[0] = 0.0
    dP_ref = [ ]
    dP_ref.append(float(0.0))
    dpstr_ref = [ ]
    dpstr_ref.append('0.0')
    for k, directory in enumerate(ref_DIR):
        dpstr_ref.append(directory.rsplit("_",1)[1])
        dP_ref.append(float(dpstr_ref[k+1]))
    dPref,dpstrref = zip(*sorted(zip(dP_ref,dpstr_ref)))
    
    print('ref ',dPref, dpstrref)
    for k in range(num_ref):
        temp = np.loadtxt('./ref_shapes/simulations_dP_'+dpstrref[k+1]+'/xprof')        
        print('./ref_shapes/simulations_dP_'+dpstrref[k+1]+'/xprof')
        vel_ref_mean[k+1] = np.mean(temp[:,6])
        
    
    #Compute for reference shape
    x_ref = vel_ref_mean[0:8]
    y_ref = dPref[0:8]
    
    #temp_ref = (np.hstack((x_ref[:]**2, x_ref[:],np.zeros_like(x_ref[:]))))
    #print(np.shape(temp_ref), np.shape(y_ref))
    #p_ref = np.linalg.lstsq(temp_ref,y_ref)[0]
    #x_fit_ref = np.linspace(vel_ref_mean[0],vel_ref_mean[6],25)
    #y_fit_ref = p_ref[0]*x_fit_ref**2 + p_ref[1]*x_fit_ref
    p_ref = (y_ref[2]-y_ref[1])/(x_ref[2]-x_ref[1])
    print("pref",p_ref)
    #################
    #####
    #dP_ref = np.array([0, 1e-9, 2e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7 ])
    dP_ = []
    dP_.append(float(0.0))
    dpstr_ = []
    dpstr_.append('0.0')
    for k, directory in enumerate(DIR_):
        dpstr_.append(directory.rsplit("_",1)[1])
        dP_.append((float(dpstr_[k+1])))
    dP= sorted(dP_)
    print(dP)
    print(vel_mean[id[0],:])
    pl.figure(figsize=(12,6))
    pl.subplot(121)
    color = ['r.','g.','b.','c.']
    line = ['r-','g-','b-','c-']
    pl.plot(vel_ref_mean[0:7]*rho*L/mu,dPref[0:7],'ko-',label='Circle')
    #pl.plot(x_fit_ref*rho*L/mu,y_fit_ref,'k--',linewidth=2)
    for j in range(len(id)):
        
        pl.plot(vel_mean[id[j],0:6]*rho*L/mu, dP[0:6], color[j],markersize=12,label='shape '+str(j))
        x_data = np.linspace(vel_mean[id[j],0],vel_mean[id[j],5],25)
        y_fit = fit_data[id[j],0]*x_data**2 + fit_data[id[j],1]*x_data
        pl.plot(x_data*rho*L/mu,y_fit,line[j],linewidth=2)
    pl.legend(loc='lower right')
    
    pl.xlabel('Re',fontsize=18)
    pl.ylabel(r'$\partial p /\partial \rho$',fontsize=18)
    #pl.plot(velocities[id_2,:-1], dP, 'g.-')
    #pl.ylim(0,0.1)
    #pl.xlim(0,0.000002)
    files = os.listdir('shapes/coords/')
    files.sort()
    pl.subplot(243)
    
    coord_1 = np.loadtxt('shapes/coords/'+files[id[0]], delimiter=' ')
    #coord_2 = np.loadtxt('shapes/coords/'+files[id_2], delimiter=' ')
    img = pl.imread('simulations_dP_1e-8/'+files[id[0]]+'/velocity.png')
    pl.imshow(img,extent=[-0.5,0.5,-0.5,0.5],cmap='gray')
    pl.plot(coord_1[:,0],coord_1[:,1],'r-',linewidth=2)
    #pl.plot(coord_2[:,0],coord_2[:,1],'g-',linewidth=2)
    a=pl.axis('equal')
    pl.axis('off')
    pl.xticks([])
    pl.title('Shape 0')
    pl.subplot(244)
    
    coord_1 = np.loadtxt('shapes/coords/'+files[id[1]], delimiter=' ')
    #coord_2 = np.loadtxt('shapes/coords/'+files[id_2], delimiter=' ')
    img = pl.imread('simulations_dP_1e-8/'+files[id[1]]+'/velocity.png')
    pl.imshow(img,extent=[-0.5,0.5,-0.5,0.5],cmap='gray')
    pl.plot(coord_1[:,0],coord_1[:,1],'r-',linewidth=2)
    #pl.plot(coord_2[:,0],coord_2[:,1],'g-',linewidth=2)
    a=pl.axis('equal')
    pl.axis('off')
    pl.xticks([])
    pl.title('Shape 1')
    pl.subplot(247)
    
    coord_1 = np.loadtxt('shapes/coords/'+files[id[2]], delimiter=' ')
    #coord_2 = np.loadtxt('shapes/coords/'+files[id_2], delimiter=' ')
    img = pl.imread('simulations_dP_1e-8/'+files[id[2]]+'/velocity.png')
    pl.imshow(img,extent=[-0.5,0.5,-0.5,0.5],cmap='gray')
    pl.plot(coord_1[:,0],coord_1[:,1],'r-',linewidth=2)
    #pl.plot(coord_2[:,0],coord_2[:,1],'g-',linewidth=2)
    a=pl.axis('equal')
    pl.axis('off')
    pl.xticks([])
    pl.title('Shape 2')
    pl.subplot(248)
    
    coord_1 = np.loadtxt('shapes/coords/'+files[id[3]], delimiter=' ')
    #coord_2 = np.loadtxt('shapes/coords/'+files[id_2], delimiter=' ')
    img = pl.imread('simulations_dP_1e-8/'+files[id[3]]+'/velocity.png')
    pl.imshow(img,extent=[-0.5,0.5,-0.5,0.5],cmap='gray')
    pl.plot(coord_1[:,0],coord_1[:,1],'r-',linewidth=2)
    #pl.plot(coord_2[:,0],coord_2[:,1],'g-',linewidth=2)
    a=pl.axis('equal')
    pl.axis('off')
    pl.xticks([])
    pl.title('Shape 3')
    
    pl.savefig("sample_shapes.jpg",dpi=300)
                
    
        
