import caffe
import numpy as np
import scipy as sp
from scipy.optimize import least_squares





# Here the transformer is defined
def polarTransformFunc(im_i):



    ########################
    # Track Reconstruction #
    ########################


    ### Initialization

    im = im_i

    # Get dimensions
    width = im.shape[1]
    height = im.shape[0]

    # Cut unnecessary far interactions (track reconstruction only)
    im_tracking = np.delete(im, np.s_[width/2:width], axis=1)

    # Find nonblack pixels (nonzero elements)
    x_arr = np.nonzero(im_tracking)[1]
    y_arr = np.nonzero(im_tracking)[0]

    # Define initial parameters for fitting (0 deg incline, at the centre of the pic)
    x0 = np.array([0, height/2])

    # Define fitting function (linear function k*x+b)
    def fun(x, t, y):
        return x[0]*t + x[1] - y


    ### Fitter

    def fitFunc(x_arr_i, y_arr_i, fun, x0, mul):

        # Use least_squares from scipy.optimize with 'cauchy' (log) robustness
        fit = sp.optimize.least_squares(fun, x0, loss='cauchy', f_scale=1,
        	args=(x_arr_i, y_arr_i)) 

        # Get k and b (see def func)
        k = np.asscalar(fit.x[0])
        b = np.asscalar(fit.x[1])

        #k = 0.1
        #b = 0.1

        # Get residuals and their standard dev 
        f_res = fit.fun
        sigma = np.std(f_res)   # squared

        # Cut outliers
        x_arr_o = np.delete(x_arr_i, np.where(f_res > mul * np.sqrt(sigma)))
        y_arr_o = np.delete(y_arr_i, np.where(f_res > mul * np.sqrt(sigma)))

        # Send k, b and image pixels w/o outliers 
        return [k, b, x_arr_o, y_arr_o]


    ### Fit Iterations

    mul = 1    # multiplier for mul*sigma cut
    num_iters = 2   # number of fitting iterations
    x_arr_pass = x_arr
    y_arr_pass = y_arr

    # Iterations
    for iter in range(0, num_iters):
        [k, b, x_arr_pass, y_arr_pass] = fitFunc(x_arr_pass, y_arr_pass, fun, x0, mul)

    #print('1 DONE')



    ###################
    # Find The Vertex #
    ###################


    ### Defining The Vertex

    # Shift image if b is too big or too small (happens with cosmic background events)
    if b > 79:
        deltaX = int(round((b-79)/k))
        im = np.roll(im, deltaX, axis=1)
        im[...,width+deltaX-1:width] = 0
        b = 79
    elif b < 0:
        deltaX = int(round(b/k))
        im = np.roll(im, deltaX, axis=1)
        im[...,width+deltaX-1:width] = 0
        b = 0

    # Initialize vertex coords as (0, b)
    vertexX = 0
    vertexY = int(round(b))

    # p1 and p2 are some points on the line k*x+b, p3 is a pixel being measured
    p1 = np.array([0, b])
    p2 = np.array([1, k+b])
    p3 = np.array([0, 0])

    # Sort arrays of pixels with according to x coordinate
    x_arr_sorted = np.sort(x_arr)
    y_arr_sorted = np.array([x for _,x in sorted(zip(x_arr, y_arr))])

    # Find norm distances from p3 to k*x+b in order to define the vertex point 
    i = 0
    while i < len(x_arr_sorted):

        # Check only first 8 x pixels
        if x_arr_sorted[i] >= 12:
            break

        # Pick the coordinates of the pixel
        p3[0] = x_arr_sorted[i]
        p3[1] = y_arr_sorted[i]

        # Check if the pixel is near enough to the line
        if np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) < 0.5:
            vertexX = x_arr_sorted[i]
            vertexY = int(k*x_arr_sorted[i] + b)
            break
        i += 1

    #print('2 DONE')



    ########################
    # Polar Transformation #
    ########################


    ### Init im_pol And Scalings

    sw = 1  # 1 if 180, 2 if 360

    # If 180 is chosen, x will start running from the vertex point (we cannot observe what is "behind" the vertex)
    if sw == 1:
        x_start = vertexX
        x_arr = np.delete(x_arr, np.where(x_arr <= x_start))
    elif sw == 2:
        x_start = 0 

    # Size of the output image
    width_pol = width
    height_pol = height

    # Define max length of a track to scale to width_rad 
    maxradius = np.sqrt(width**2 + height**2)

    # Initialize a blank pic
    im_pol = np.zeros((height_pol, width_pol))

    # Find scalings
    rscale = float(width_pol) / maxradius
    tscale = float(height_pol) / (sw*180 + 1)


    ### Transformation

    for y in y_arr:
        dy = y - vertexY    # calculate from the vertex
        for x in x_arr:
            dx = x - vertexX    # calculate from the vertex
            t = ((np.arctan2(dy,dx)*180)/np.pi) + sw*90 # find angle
            r = np.sqrt(dx**2 + dy**2)    # find radius
            t_sc = int(np.floor(t*tscale))
            r_sc = int(np.floor(r*rscale))
            im_pol[t_sc, r_sc] = im[y, x]    # write with respect to the scaling parameters


    ### Return im_pol

    #print('3 DONE')
    
    im_pol = im_pol.astype(np.uint8)
    return im_pol 



# This class is for Caffe Framework. It still bottlenecks the entire system. Maybe PyCUDA could solve this issue...
class PolarTransform(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert bottom[0].data.ndim >= 3,    'requires image data'
        assert len(top) == 1,               'requires a single layer.top'

    def reshape(self, bottom, top):

        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape) #This is required!

    def forward(self, bottom, top):

        # Copy to top blob(s):
        top[0].data[...] = bottom[0].data # This is correctly copying data

        pic_batch = top[0].data #OK! This works!
        #print(bottom[0].data.shape)

        im = np.zeros((80, 100))
        # Loop over events and transform
        for ievt in range(0, np.size(pic_batch, 0)):
            #pic_batch[ievt,0,...] = polarTransformFunc(pic_batch[ievt,0,...])
            #pic_batch[ievt,0,...] = np.transpose(pic_batch[ievt,0,...])
            for ii in range(0, 80):
                for jj in range(0, 100):
                    im[ii, jj] = pic_batch[ievt, 0, jj, ii]
            #print(im.shape)
            im = polarTransformFunc(im)
            for ii in range(0,80):
                for jj in range(0,100):
                    pic_batch[ievt, 0, jj, ii] = im[ii, jj]

        # Pump back into top
        top[0].data[...] = pic_batch
        #top[0].data[...] = bottom[0].data   # control

    def backward(self, top, propagate_down, bottom):
        pass
