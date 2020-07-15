import numpy as np
import pdb

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import rc
from matplotlib import rcParams
rc('text', usetex = True)
rcParams['font.size']=18
rcParams['font.sans-serif'] = "Comic Sans MS"
rcParams['font.family']='sans-serif'

def aDist(j, a):
    alpha = np.sqrt(np.pi)/(3*np.sqrt(2))
    beta = (4./3)*(4 - np.pi)/96

    alpha2 = np.sqrt(np.pi)/3
    beta2 = (4 - np.pi)/12

    dl = 1
    for i in np.arange(1,j+1):
        totA = np.zeros((i, len(a)))
        jointDist = np.zeros((i, len(a)))
        l = np.arange(1, i+1, dl)
        fl = l**(-3/2)
        norm = np.sum(fl)
        fl/=norm
        #print(norm)

        lam2 = alpha2**3/beta2*(j+1)**(9/2)/(j)**(3)
        U2 = alpha2*(j+1)**(3/2)
        invGauss2 = np.sqrt(lam2/(2*np.pi*a**3))*np.exp(-lam2*(a-U2)**2/(2*U2**2*a))
        K=0
        for k in l:
            lam = alpha**3/beta*(k)**(9/2)/(k-2)**(3)
            U = alpha*(k)**(3/2)
            if k<3:
                invGauss = np.zeros(len(a))
                invGauss[int(k)] = 1
            else:
                invGauss = np.sqrt(lam/(2*np.pi*a**3))*np.exp(-lam*(a-U)**2/(2*U**2*a))
            jointDist[K,:] = (1 - j**(-1/2))*fl[K]*invGauss
            K+=1
            #pdb.set_trace()
        fA = np.sum(jointDist, axis = 0) + j**(-1/2)*invGauss2
        Fa = np.cumsum(fA)*(a[1]-a[0])
        totA[i-1,:] = fA

    fArea = np.sum(totA, axis=0)/j
    Farea = np.cumsum(fArea)
    #pdb.set_trace()
    return(fArea, Farea, fA, Fa)

def inverseSample(F, samp, a):
    aSample = []
    for i in samp:
        ind = np.where((i- Fa)**2 == np.min((i-Fa)**2))
        aSample = np.append(aSample, a[ind[0][0]])
    return(aSample)

numRills = 100
L = np.arange(10,110,10)
#color=iter(cm.gray(np.linspace(0,1,3/2*len(L))))

dl = 1.
r = 0.5
da = dl*r

rho = 1000
g = 9.8
S = 0.1
m = 1/3#exponent for rill width
m2 = 1/(1-m)
R = 0.05/3600 #cm/hour to m/s
w0 =0.01
manningsN = 0.075
cH = (12/15)**(2/3)*manningsN/S**(1/2)
K = 1.0

l0 = np.arange(10, 110, 10)
gamma = np.linspace(0.5, 2.25, 10)
omegaTot = []
tauTot = []
concTot = []
totA = []
lVec = []
count = 0

concMat = np.zeros(len(gamma))
tauMat = np.zeros(len(gamma))
omegaMat = np.zeros(len(gamma))
W=0
I=0
for i in l0:
    omegaTot = []
    tauTot = []
    concTot = []
    lVec=[]
    for j in L:
        num = 0
        numRills = 100
        while num<10:
            a = np.arange(1,j**2)
            fATot, FaTot, fA, Fa = aDist(j, a)
            samp = np.random.uniform(0, 1, numRills*j)
            aSample = inverseSample(FaTot, samp, a)
            aSample2 = inverseSample(Fa, samp, a)
            Q = da*aSample*R
            #wR = K*Q**m
            #h = cH*Q/wR**(5/3)
            #omega = Q/wR
            wR = w0 + K*((Q + da*R)**(1 + m) - Q**(1+m))/((1 + m)*R*da)
            omega = (Q + R*da/2)/wR
            h = cH*(Q+da*R/2)/wR**(5./3.)

            tau = 0.0001*(h*S*g*rho)**gamma[I]*wR*dl
            omega= 0.0001*(omega*S*g*rho)**gamma[I]*wR*dl

            Q2 = da*aSample2*R
            conc = (1 - np.exp(-aSample2[:numRills+1]*da/i))*Q2[:numRills+1]

            totA = np.append(totA, np.sum(aSample[:numRills+1])/(j*numRills))

            omegaTot = np.append(omegaTot, np.sum(omega))
            tauTot = np.append(tauTot, np.sum(tau))
            concTot = np.append(concTot, np.sum(conc))
            lVec =np.append(lVec, j)
            num+=1
            if I==3:
                plt.figure(1)
                plt.plot(j, np.sum(omega), 'ok', label = '$\\omega')
                #plt.grid()
                plt.ylabel('$\\Sigma \\omega$ [M T$^{-3}$]')
                plt.xlabel('$L$ [L]')


                plt.plot(j, np.sum(tau), '+k', label = '$\\tau$')
                #plt.grid()
                plt.ylabel('$\\Sigma \\tau$ [M L$^{-1}$ T$^{-2}$]')
                plt.xlabel('$L$ [L]')

                plt.plot(j,np.sum(conc), 'dk', label = 'c')
                #plt.grid()
                plt.ylabel('$\\Sigma Q_{s}$ [L$^{2}$ T$^{-1}$]')
                plt.xlabel('$L$ [L]')

                if num==10:
                    plt.legend()
                    plt.yscale('log')
                    plt.xscale('log')
                    plt.grid()
    tempC = np.polyfit(np.log(lVec), np.log(concTot), 1)
    tempO = np.polyfit(np.log(lVec), np.log(omegaTot), 1)
    tempT = np.polyfit(np.log(lVec), np.log(tauTot), 1)
    concMat[I] = tempC[0]
    omegaMat[I] = tempO[0]
    tauMat[I] = tempT[0]
    I+=1

    W+=1
    #print(w)

#pdb.set_trace()
fig, ax1 = plt.subplots()

ax1.plot(l0, concMat, 'dk', label = '$c$')
ax1.set_ylabel('$\\beta$ in $Q_{s} \propto L^{\\beta}$' )
ax1.set_xlabel('$\\kappa/T_{c}$ in $c = T_{c}(1 - e^{\kappa/T_{c}a})$')
ax2 = ax1.twiny()
ax2.plot(gamma, tauMat, '+k', label = '$\\tau$')
ax2.plot(gamma, omegaMat, 'ok', label = '$\\omega$')
ax2.set_xlabel('$\eta$ in $D_{s} \propto \\tau^{\eta}, \\omega^{\eta}$')
plt.grid()
#ax1.set_yscale('log')
#ax2.set_yscale('log')
#ax1.set_xscale('log')
#ax2.set_xscale('log')
plt.legend()


#X1, Y1 = np.meshgrid(gamma, m)
#X2, Y2 = np.meshgrid(l0, m)

#pdb.set_trace()

#levels = np.arange(1.0, 10.0, 0.25)
#plt.figure()

#CS = plt.contourf(Y2, X2, concMat, levels)
#plt.clabel(CS, inline=1, fontsize=12)
#plt.colorbar()

#plt.figure()
#CS = plt.contourf(Y1, X1, tauMat, levels)
#plt.clabel(CS, inline=1, fontsize = 12)
#plt.colorbar()

#plt.figure()
#CS = plt.contourf(Y1, X1, omegaMat, levels)
#plt.clabel(CS, inline=1, fontsize=12)
#plt.colorbar()

plt.show(block = False)
pdb.set_trace()
