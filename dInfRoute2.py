import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, linalg
import pdb

from matplotlib import rc
from matplotlib import rcParams
#
#rcParams['font.size']=18
#rcParams['font.sans-serif'] = "Comic Sans MS"
#rcParams['font.family']='sans-serif'

def Scheidegger(lY, lX, interRill, ang):
    import numpy as np
    mask = np.zeros((lY,lX)) #Initialize mask
    add = np.array([0, 1])
    rillPos = np.arange(int(interRill/2), lY, interRill) #initialize vector of occupied rillsites

    temp = np.copy(rillPos)
    k = 0
    hNew = interRill/2.
    h=0
    hOld = 0.0
    i=0
    hMax = interRill/(2*ang)
    while i <lX: #Loop through all positions
        if hNew/(interRill/2)==1:    #when position is a multiple of interRill, initialize new rills to maintain spacing
            if np.mod(k, 2)==0:
                rillPos = np.arange(interRill/4, lY, interRill)
            else:
                rillPos = np.arange(3*interRill/4, lY, interRill)
            lenHolder = len(rillPos)
            rillPos2 = []
            for s in rillPos:
                rillPos2 = np.append(rillPos2,s+add)
            rillPos = rillPos2
            rillPos = rillPos.astype('int')
            #mask[rillPos, i] = 1   #add 1 where rill is
            steps = np.random.randint(0, 2, size= lenHolder) #Such topography differs for topography considered for theory, which assumes a rectangular cross-channel profile. However, values of discharge will be the same.step left-right, or up-down
            steps[steps==0] = -1    #adjust zeros to be left or up
            steps = np.tile(steps, (len(add),1)).ravel('F')
            k += 1
            mask[rillPos, i] = 1
            h=0
            hOld = 0.
            i+=1
            #hNew = np.nan

        h+=1
        hTemp = h*ang
        hNew = np.round(hTemp)
        if h==hMax:
            #i+=1#pdb.set_trace()
            continue
        if hNew==hOld:
            #rillPos[rillPos>lY-1] = rillPos[rillPos>lY-1] - (lY)
            mask[rillPos, i] =1
            hOld = hNew
            i+=1
        else:
            rillPos = rillPos + steps  #rill position is adjusted with valid steps
            rillPos[rillPos>lY-1] = rillPos[rillPos>lY-1] - (lY)
            mask[rillPos, i] = 1
            hOld = hNew
            i+=1

    return(mask)
def topoMake(mask, T, slp,dx):
    lY, lX = np.shape(mask)

    D = 0.0001
    U = 0.1
    dt =0.01*(dx**2.0)/D

    x = np.arange(0, lX)
    y = np.arange(0, lY)

    X, Y = np.meshgrid(x, y)
    z = np.ones((lY,lX))
    ind = np.arange(0,lY*lX)

    diri = 1 - mask
    diri = np.ravel(diri, 'F')

    bound = np.ones_like(z)
    #bound[0,:] = 2
    #bound[-1,:] = 3
    bound = np.ravel(bound, 'F')

    diagVals = (D*dt/(dx**2.0))*np.ones((lY*lX))
    diagValsU1 = np.copy(diagVals)
    diagValsU2 = np.copy(diagVals)
    diagValsL1 = np.copy(diagVals)
    diagValsL2 = np.copy(diagVals)

    #diagValsU1[bound==2] = 0#2.0*D*dt/(dx**2.0)
    #diagValsL1[bound==2] = 2.0*D*dt/(dx**2.0)
    #diagValsU1[bound==3] = 2.0*D*dt/(dx**2.0)
    #diagValsL1[bound==3] = 0#2.0*D*dt/(dx**2.0)

    bound = np.reshape(bound, (lY, lX), 'F')
    bound[:,0] = 4
    bound[:,-1] = 5
    bound[0,:] = 2
    bound[-1,:] = 3
    bound = np.ravel(bound, 'F')

    indAdj3 = np.zeros_like(bound)
    indAdj2 = np.zeros_like(bound)
    indAdj3[bound==3] = int(lY-1)
    indAdj2[bound==2] = int(-lY+1)

    diagValsU2[bound==4] = 2.0*D*dt/(dx**2.0)
    diagValsL2[bound==4] = 0.0#2.0*D*dt/(dx**2.0)
    diagValsL2[bound==5] = 2.0*D*dt/(dx**2.0)
    diagValsU2[bound==5] = 0.0#2.0*D*dt/(dx**2.0)

    bound[bound!=1] = 0.0

    midVals = (-4.0*D*dt/(dx**2.0) + 1.0)*np.ones((lY*lX))
    midVals[diri==0.0] = 1.0
    diagValsU1[diri==0.0] = 0.0
    diagValsL1[diri==0.0] = 0.0
    diagValsU2[diri==0.0] = 0.0
    diagValsL2[diri==0.0] = 0.0

    rows = np.concatenate((ind, ind[:-1], ind[1:], ind[:-lY], ind[lY:]))
    cols = np.concatenate((ind, ind[1:]+indAdj2[1:], ind[:-1]+indAdj3[:-1], ind[lY:], ind[:-lY]))

    vals = np.concatenate((midVals, diagValsU1[:-1], diagValsL1[1:], diagValsU2[:-lY], diagValsL2[lY:]))

    H = coo_matrix((vals, (rows, cols)), shape =(lY*lX, lY*lX)).tocsr()

    t = 0

    z = z.ravel('F')
    uplift = U*np.ones_like(z)*dt
    #uplift[bound==0] = 0.0
    uplift[diri==0.0] = 0.0
    #z+=uplift
    #pdb.set_trace()
    while t<T:
        z = H*z+uplift
        t = t+1
    z = z.reshape((lY, lX), order = 'F')

    #pdb.set_trace()
    z = z - slp*X

    return(z)
def hydroCorrect(z):
    fillIncrement = 0.001
    lY, lX = np.shape(z)
    slope = slopeSimple(z)
    zVec = z.ravel('F')
    #sVec = slope.ravel('C')
    k = 0
    while any(slope.ravel()<=0.000):
        ind = np.asarray(np.where(slope<=0.000))
        indJ = ind[1]
        indI = ind[0]
        num = len(indI)
        #indJ = np.floor(ind/lY).astype('int')
        #indI = np.mod(ind,lY)
        print(len(indI))
        #zMin = z[indI, indJ]
        n = 0
        while n<num:
            i, j = indI[n], indJ[n]
            print(i,j)
            print(z[i,j])
            #pdb.set_trace()
            zTemp = z[i-1:i+2, j-1:j+2]
            zTemp = np.delete(zTemp, (1,1))
            minZ = np.min(zTemp)
            z[i, j] = minZ+fillIncrement
            n+=1
            print(z[i,j], 'New Round')



        #zVec[sVec==0]+=fillIncrement
        #zTemp = zVec.reshape((lY, lX), order = 'F')
        slope= slopeSimple(z)
        sVec = slope.ravel('C')
        k+=1
        if k>100:
            print('Max Fill-Increment -- moving on')
            break
    print(k)
    z = zVec.reshape((lY, lX), order = 'F')
    return(z)
def slopeSimple(z):
    lY, lX = np.shape(z)
    y = np.arange(0, lY)
    x = np.arange(0, lX)
    slope = np.zeros_like(z)
    slp = np.zeros(8)
    s2 = np.sqrt(2.)
    #fdir = np.zeros_like(z)
    ind= np.arange(0, 8)

    for i in y:
        for j in x:
            if i>0 and i<lY-1 and j>0 and j<lX-1:
                slp[0] = z[i,j] - z[i,j+1]
                slp[1] = (z[i,j] - z[i-1,j+1])/s2
                slp[2] = z[i,j] - z[i-1,j]
                slp[3] = (z[i,j] - z[i-1,j-1])/s2
                slp[4] = z[i,j] - z[i,j-1]
                slp[5] = (z[i,j] - z[i+1,j-1])/s2
                slp[6] = z[i,j] - z[i+1,j]
                slp[7] = (z[i,j] - z[i+1,j+1])/s2

            elif i==0 and j>0 and j<lX-1:
                slp[0] = z[i,j] - z[i,j+1]
                slp[1] = 0.01
                slp[2] = 0.01
                slp[3] = 0.01
                slp[4] = z[i,j] - z[i,j-1]
                slp[5] = (z[i,j] - z[i+1,j-1])/s2
                slp[6] = z[i,j] - z[i+1,j]
                slp[7] = (z[i,j] - z[i+1,j+1])/s2
            elif i==lY-1 and j>0 and j<lX-1:
                slp[0] = z[i,j] - z[i,j+1]
                slp[1] = (z[i,j] - z[i-1,j+1])/s2
                slp[2] = z[i,j] - z[i-1,j]
                slp[3] = (z[i,j] - z[i-1,j-1])/s2
                slp[4] = z[i,j] - z[i,j-1]
                slp[5] = 0.01#(z[i,j] - z[0,j-1])/s2
                slp[6] = 0.01#z[i,j] - z[0,j]
                slp[7] = 0.01#(z[i,j] - z[0,j+1])/s2

            slope[i,j] = np.max(slp)

            if i==0 or i==lY-1 and j==0:
                slope[i,j] = 0.01

            elif j==lX-1:
                slope[i,j] = 0.1
    return(slope)
def dInfSlope(z, i, j, dx):
    #D-infinity slope routine

    lY, lX = np.shape(z)
    slope = 0
    angAdj = np.array([0, np.pi/2, np.pi/2., np.pi, np.pi, 6.*np.pi/4., 6.*np.pi/4., 2*np.pi])
    sign = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    indHolder = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    count=0
    iNeighbors = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    jNeighbors = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    slope = np.zeros(8)
    xSlope = np.zeros(8)
    if i>0 and i<lY-1 and j>0 and j<lX-1:
        #Rotate around grid point counter clockwise for dINF
        slope[-1] = z[i,j] - z[i,j+1]
        slope[0] = z[i,j] - z[i,j+1]
        slope[1:3] = z[i,j] - z[i-1,j]
        slope[3:5] = z[i,j] - z[i,j-1]
        slope[5:7] = z[i,j] - z[i+1,j]

        xSlope[0] = z[i,j+1] - z[i-1,j+1]
        xSlope[1] = z[i-1,j] - z[i-1,j+1]
        xSlope[2] = z[i-1,j] - z[i-1,j-1]
        xSlope[3] = z[i,j-1] - z[i-1,j-1]
        xSlope[4] = z[i,j-1] - z[i+1,j-1]
        xSlope[5] = z[i+1,j] - z[i+1,j-1]
        xSlope[6] = z[i+1,j] - z[i+1,j+1]
        xSlope[7] = z[i,j+1] - z[i+1,j+1]

        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
        #pdb.set_trace()
    elif i==0 and j>0 and j<lX-1:
        slope[-1] = z[i,j] - z[i,j+1]
        slope[0] = z[i,j] - z[i,j+1]
        #slope[1:3] = 0#z[i,j] - z[i-1,j]
        slope[1:3] = z[i,j] - z[-1, j] #for periodic boundary conditions
        slope[3:5] = z[i,j] - z[i,j-1]
        slope[5:7] = z[i,j] - z[i+1,j]

        xSlope[4] = z[i,j-1] - z[i+1,j-1]
        xSlope[5] = z[i+1,j] - z[i+1,j-1]
        xSlope[6] = z[i+1,j] - z[i+1,j+1]
        xSlope[7] = z[i,j+1] - z[i+1,j+1]

        #PERIODIC BOUNDARY CONDITIONS
        xSlope[0] = z[i,j+1] - z[-1,j+1]
        xSlope[1] = z[-1,j] - z[-1,j+1]
        xSlope[2] = z[-1,j] - z[-1,j-1]
        xSlope[3] = z[i,j-1] - z[-1,j-1]

        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
    elif i==lY-1 and j>0 and j<lX-1:
        slope[-1] = z[i,j] - z[i,j+1]
        slope[0] = z[i,j] - z[i,j+1]
        slope[1:3] = z[i,j] - z[i-1,j]
        slope[3:5] = z[i,j] - z[i,j-1]
        #slope[5:7] = z[i,j] - z[i+1,j]
        #PERIODIC CONDITIONS
        slope[5:7] = z[i,j] -z[0,j]

        xSlope[0] = z[i,j+1] - z[i-1,j+1]
        xSlope[1] = z[i-1,j] - z[i-1,j+1]
        xSlope[2] = z[i-1,j] - z[i-1,j-1]
        xSlope[3] = z[i,j-1] - z[i-1,j-1]

        iNeighbors[iNeighbors==1] = -lY+1
        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
        #PERIODIC CONDITIONS
        xSlope[4] = z[i,j-1] - z[0,j-1]
        xSlope[5] = z[0,j] - z[0,j-1]
        xSlope[6] = z[0,j] - z[0,j+1]
        xSlope[7] = z[i,j+1] - z[0,j+1]
    elif i>0 and i<lY-1 and j==0:
        slope[-1:1] = z[i,j] - z[i,j+1]
        slope[1:3] = z[i,j] - z[i-1,j]
        #slope[3:5] = z[i,j] - z[i,j-1]
        slope[5:7] = z[i,j] - z[i+1,j]
        #PERIODIC BOUNDARY
        #slope[-1:1] - z[i,j] - z[i,-1]

        xSlope[0] = z[i,j+1] - z[i-1,j+1]
        xSlope[1] = z[i-1,j] - z[i,j+1]

        xSlope[6] = z[i+1,j] - z[i+1,j+1]
        xSlope[7] = z[i,j+1] - z[i+1,j+1]

        iNeighbors[jNeighbors==-1] = 0
        jNeighbors[jNeighbors==-1] = 0
        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
        #PERIODIC BOUNDARY CONDITIONS
        #xSlope[2] = z[i-1,j] - z[i-1,-1]
        #xSlope[3] = z[i,-1] - z[i-1,-1]
        #xSlope[4] = z[i,-1] -z[i+1,-1]
        #xSlope[5] = z[i+1,j] - z[i+1,-1]
    elif i>0 and i<lY-1 and j==lX-1:
        #slope[-1:1] = z[i,j] - z[i,j+1]
        slope[1:3] = z[i,j] - z[i-1,j]
        slope[3:5] = z[i,j] - z[i,j-1]
        slope[5:7] = z[i,j] - z[i+1,j]
        slope[0] = 0.1
        slope[-1] = 0.1
        #PERIODIC BOUNDARY CONDITIONS
        #slope[-1:1] = z[i,j] - z[i,0]

        xSlope[2] = z[i-1,j] - z[i-1,j-1]
        xSlope[3] = z[i,j-1] - z[i-1,j-1]
        xSlope[4] = z[i,j-1] - z[i+1,j-1]
        xSlope[5] = z[i+1,j] - z[i+1,j-1]

        iNeighbors[jNeighbors==1] = 0
        jNeighbors[jNeighbors==1] = 0
        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
        #PERIODIC BOUNDARY CONDITIONS
        #xSlope[0] = z[i,0] - z[i-1,0]
        #xSlope[1] = z[i-1,j] - z[i-1,0]
        #xSlope[6] = z[i+1,j] - z[i+1,0]
        #xSlope[7] = z[i,0] - z[i+1,0]
    elif i==0 and j==0:
        slope[-1:1] = z[i,j] - z[i,j+1]
        slope[5:7] = z[i,j] - z[i+1,j]

        xSlope[6] = z[i+1,j] - z[i+1,j+1]
        xSlope[7] = z[i,j+1] - z[i+1,j+1]

        iNeighbors[jNeighbors==-1] = 0
        jNeighbors[iNeighbors==-1] = 0
        jNeighbors[jNeighbors==-1] = 0
        iNeighbors[iNeighbors==-1] = 0

        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
    elif i==0 and j==lX-1:
        slope[3:5] = z[i,j] - z[i,j-1]
        slope[5:7] = z[i,j] - z[i+1,j]

        xSlope[4] = z[i,j-1] - z[i+1,j-1]
        xSlope[5] = z[i+1,j] - z[i+1,j-1]

        iNeighbors[jNeighbors==1] = 0
        jNeighbors[iNeighbors==-1]= 0
        jNeighbors[jNeighbors==1] = 0
        iNeighbors[iNeighbors==-1] = 0

        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
    elif i==lY-1 and j==0:
        slope[-1:1] = z[i,j] - z[i,j+1]
        slope[1:3] = z[i,j] - z[i-1,j]

        xSlope[0] = z[i,j+1] - z[i-1,j+1]
        xSlope[1] = z[i-1,j] - z[i-1,j+1]

        iNeighbors[jNeighbors==-1] = 0
        jNeighbors[iNeighbors==1] = 0
        jNeighbors[jNeighbors==-1] = 0
        iNeighbors[iNeighbors==1] = 0

        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]
    elif i==lY-1 and j==lX-1:
        slope[1:3] = z[i,j] - z[i-1,j]
        slope[3:5] = z[i,j] - z[i,j-1]

        xSlope[2] = z[i-1,j] - z[i-1,j-1]
        xSlope[3] = z[i,j-1] - z[i-1,j-1]

        iNeighbors[jNeighbors==1] = 0
        jNeighbors[iNeighbors==1] = 0
        jNeighbors[jNeighbors==1] = 0
        iNeighbors[iNeighbors==1] = 0

        dz = z[i,j] - z[i+iNeighbors,j+jNeighbors]

    slope[slope<0] = 0
    xSlope[xSlope<0] = 0
    xSlope[slope<0] = 0

    dz[np.array([1,3,5,7])]/=np.sqrt(2)
    try:
        ind1 = np.where(dz==np.max(dz))[0][0]
        temp = np.array([dz[ind1-1],dz[np.mod(ind1+1,8)]])
    except:
        pdb.set_trace()

    if temp[0]>=temp[1]:
        ind = ind1
    elif temp[1]>temp[0]:
        ind = ind1-1

    theta = np.arctan(xSlope/slope)
    theta[theta>np.pi/4.] = np.pi/4.

    try:
        slp = np.sqrt(slope[ind]**2. + xSlope[ind]**2.)/np.sqrt(2.)
        angle = angAdj[ind] + sign[ind]*theta[ind]
        IND = indHolder[ind]
        slp = slp/dx
    except:
        tempInd = np.where(ind==True)[0]
        slp = np.sqrt(slope[tempInd[0]]**2. + xSlope[tempInd[0]]**2.)/np.sqrt(2.)
        angle= angAdj[tempInd[0]] + sign[tempInd[0]]*theta[tempInd[0]]
        slp = slp/dx
    return(slp, angle)
def dInfFlowRoute(angle, i, j):

    piOver4 = np.pi/4.

    flow0 = 0
    flow1 = 0
    flow2 = 0
    flow3 = 0
    flow4 = 0
    flow5 = 0
    flow6 = 0
    flow7 = 0

    temp = np.floor(angle/piOver4)
    if temp == 0:
        flow1 = (angle)/piOver4
        flow0 = (piOver4 - angle)/piOver4

    elif temp ==1:
        flow1 = (2*piOver4 - angle)/piOver4
        flow2 = (angle - piOver4)/piOver4

    elif temp ==2:
        flow2 = (3*piOver4 - angle)/piOver4
        flow3= (angle - 2*piOver4)/piOver4

    elif temp ==3:
        flow3 = (4*piOver4 - angle)/piOver4
        flow4 = (angle - 3*piOver4)/piOver4

    elif temp ==4:
        flow5 = (angle - 4*piOver4)/piOver4
        flow4 = (5*piOver4 - angle)/piOver4

    elif temp ==5:
        flow6 = (angle - 5*piOver4)/piOver4
        flow5 = (6*piOver4 - angle)/piOver4

    elif temp ==6:
        flow7 = (angle - 6*piOver4)/piOver4
        flow6 = (7*piOver4 - angle)/piOver4

    elif temp ==7:
        flow0 = (angle - 7*piOver4)/piOver4
        flow7 = (8*piOver4 - angle)/piOver4

    elif temp ==8:
        flow0 = (9*piOver4 - angle)/piOver4
        flow1 = (angle - 8*piOver4)/piOver4
    elif temp>8:
        print('Angle greater than 2 pi')

    fDir = np.array([flow0, flow1, flow2, flow3, flow4, flow5, flow6, flow7])
    return(fDir)
def flowRout(z, runoff, nIter, dx):
    delta = dx
    manningsN = 0.03
    lY, lX = np.shape(z)
    zOrig = np.copy(z)
    iNeighbors = [0, -1, -1, -1, 0, 1, 1, 1]
    jNeighbors = [1, 1, 0, -1, -1, -1, 0, 1]
    depthOld = np.zeros_like(z)
    counts = np.zeros_like(z)
    k0=0
    err = 1
    z = hydroCorrect(z)
    sMin = 0.01
    slope = np.zeros_like(z)
    while k0<nIter:
        flow = np.ones_like(z)*runoff*dx**2
        area = np.ones_like(z)

        argZ = np.flip(np.argsort(z.ravel('F')))
        for k in argZ:
            i = np.mod(k, lY, dtype = 'int')
            j = np.floor(k/lY).astype('int')

            S, Theta = dInfSlope(z, i, j,dx)
            if S<=sMin:
                S = sMin
            slope[i,j] = S
            fDir = dInfFlowRoute(Theta, i, j)

            inds = np.where(fDir>0)[0]
            counts[i,j] = len(inds)

            for kk in inds:
                if j+jNeighbors[kk]>lX-1:
                    continue
                    print('Bottom')
                elif i+iNeighbors[kk]<0:
                    continue
                    #area[-1, j+jNeighbors[kk]]+=area[i,j]*fDir[kk]
                    #flow[-1, j+jNeighbors[kk]]+=flow[i,j]*fDir[kk]
                elif i+iNeighbors[kk]>lY-1:
                    continue
                    #area[0, j+jNeighbors[kk]]+=area[i,j]*fDir[kk]
                    #flow[0, j+jNeighbors[kk]]+=flow[i,j]*fDir[kk]
                else:
                    area[i+iNeighbors[kk], j+jNeighbors[kk]]+=area[i,j]*fDir[kk]
                    flow[i+iNeighbors[kk], j+jNeighbors[kk]]+=flow[i,j]*fDir[kk]

        depth = (flow*manningsN/(delta*np.sqrt(slope)))**(3./5.)

        increment = (zOrig + depth-z)/nIter
        z+=increment
        z = hydroCorrect(z)
        depthTot = z - zOrig
        if any(depthTot.ravel()<0):
            pdb.set_trace()

        #pdb.set_trace()
        err = (depthTot - depthOld)/depthTot
        err = np.sum(err)/(lY*lX)
        depthOld = depthTot
        k0+=1
        print(err)
        #pdb.set_trace()
    flow = depthTot**(5./3)*delta*np.sqrt(slope)/(manningsN)
    return(depthTot, flow, area)
def nanCorrect(z):
    neighborI = [1, 0, -1, 1, -1, 1, 0, -1]
    neighborJ = [-1, -1, -1, 0, 0, 1, 1, 1]
    ind = np.where(np.isnan(z))
    for i in np.arange(0, len(ind[0])):
        zTemp = z[ind[0][i]+neighborI, ind[1][i]+neighborJ]
        aveZ = np.nanmean(zTemp)
        z[ind[0][i], ind[1][i]] = aveZ
    return(z)

z = np.load('zHydroCorr.npy')
z = nanCorrect(z)
z=z[100:600,:]
#lY, lX = 1100, 500
dx = 0.02
#mask = Scheidegger(int(lY), int(lX), 10, 0.66)
#z = topoMake(mask, 1000, 1.0, dx)
#mask = np.array(mask, dtype = 'bool')
#z[mask]-=0.03

runoff = 5/(100*3600)

depth, flow, area = flowRout(z, runoff, 50, dx)


pdb.set_trace()
