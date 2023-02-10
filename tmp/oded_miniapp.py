# import numpy as cp
# import pandas as cudf

import cupy as cp
import cudf
import time

import rmm
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.ManagedMemoryResource(),
    initial_pool_size=2**35,
    maximum_pool_size=2**35
)
rmm.mr.set_current_device_resource(pool)
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)







kmax = 11
M = 600
L = 900

IstrP = 1
IstrU = 1
IstrV = 1

IendP = L
IendU = L
IendV = L + 1

JstrP = 1
JstrU = 1
JstrV = 1


JendP = M
JendU = M + 1
JendV = M

IminS = IstrP - 3
ImaxS = IendP + 3
JminS = JstrP - 3
JmaxS = JendP + 3

print(IminS)
print(ImaxS)



iterations=100



dimR = L
dimC = M

ubar = cp.zeros([dimR, dimC, kmax])
vbar = cp.zeros([dimR, dimC, kmax])
ubarS = cp.zeros([kmax, dimR, dimC])
vbarS = cp.zeros([kmax, dimR, dimC])

aJ = cp.arange(0, dimC)
aI = cp.arange(0, dimR)

aJRep = cp.tile(aJ, (dimR, 1))
aIRep = cp.tile(aI, (dimC, 1)).transpose()

on_u = aJRep
om_v = aIRep
Drhs = on_u + om_v
ubar_stokes = aIRep
vbar_stokes = aJRep

umask_wet = cp.ones([dimR, dimC])
vmask_wet = cp.ones([dimR, dimC])
umask_wet = cp.triu(umask_wet)  # replace triu with tril if lower triangle is needed
vmask_wet = cp.triu(vmask_wet)  #

for k in range(kmax):
    ubar[:, :, k] = aIRep;
    vbar[:, :, k] = aJRep;
    ubarS[k, :, :] = aIRep;
    vbarS[k, :, :] = aJRep;

#     print(k)


# print(umask_wet)
# vmask_wet = cp.ones([M,L])


# print(on_u.shape)
# print(aI.shape)

# print(aJRep.shape)
# print(aIRep.shape)

# print(aJRep)
# print(aIRep)
# numpy.repeat()



####TAKE One - straightforward implementation.
# %%timeit


start_gpu = cp.cuda.Event()
end_gpu = cp.cuda.Event()
start_gpu.record()
start_cpu = time.perf_counter()


for it in range (iterations):
    test1=0
    test2=0
    for k in range(kmax):
        cff = 0.5*on_u[1:dimR,:]
        cff1 = cff*(Drhs[1:dimR,:] + Drhs[0:(dimR-1),:])
        DUon = ubar[1:dimR,:,k]*cff1
        cff5 = cp.sign(cp.sign(umask_wet[1:dimR,:]) - 1.0)
        cff6 = 0.5 + 0.5 *cp.sign(ubar_stokes[1:dimR,:])*umask_wet[1:dimR,:]

        cff7 = 0.5*umask_wet[1:dimR,:]*cff5 + cff6*(1.0 - cff5)
        cff1 = cff1*cff7

        DUSon = ubar_stokes[1:dimR,:]*cff1
        DUon = DUon + DUSon

        cff = 0.5*om_v[:,1:dimC]
        cff1 = cff*(Drhs[:,1:dimC] + Drhs[:,0:(dimC-1)])
        DVom = vbar[:,1:dimC,k]*cff1
        cff5 = cp.sign(cp.sign(vmask_wet[:,1:dimC]) - 1.0)
        cff6 = 0.5 + 0.5*cp.sign(vbar_stokes[:,1:dimC])*vmask_wet[:,1:dimC]
        cff7 = 0.5*vmask_wet[:,1:dimC]*cff5 + cff6*(1.0 - cff5)
        cff1 = cff1*cff7

        DVSom = vbar_stokes[:,1:dimC]*cff1
        DVom = DVom + DVSom


        test1 = test1 + cp.sum(DUon)
        test2 = test2 + cp.sum(DVom)



    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu
print(t_gpu/1000.0/iterations)
print(t_cpu/iterations)
print(test1)
print(test2)

##### Optimized
# %%timeit
start_gpu = cp.cuda.Event()
end_gpu = cp.cuda.Event()
start_gpu.record()
start_cpu = time.perf_counter()
for it in range(iterations):
    test1 = 0
    test2 = 0

    cff = 0.5 * on_u[1:dimR, :]
    umask_wetMask = umask_wet[1:dimR, :]
    cff1 = cff * (Drhs[1:dimR, :] + Drhs[0:(dimR - 1), :])
    cff5 = cp.sign(cp.sign(umask_wetMask) - 1.0)
    cff6 = 0.5 + 0.5 * cp.sign(ubar_stokes[1:dimR, :]) * umask_wetMask

    cff7 = 0.5 * umask_wetMask * cff5 + cff6 * (1.0 - cff5)
    cff1 = cff1 * cff7

    sumArrayU = cp.zeros([dimR - 1, dimC])
    DUSon = ubar_stokes[1:dimR, :] * cff1
    for k in range(kmax):
        DUon = ubarS[k, 1:dimR, :] * cff1
        #         DUon = ubar[1:dimR,:,k]*cff1
        sumArrayU = sumArrayU + DUon + DUSon

    cff = 0.5 * om_v[:, 1:dimC]
    cff1 = cff * (Drhs[:, 1:dimC] + Drhs[:, 0:(dimC - 1)])
    vmask_wetMask = vmask_wet[:, 1:dimC]
    cff5 = cp.sign(cp.sign(vmask_wetMask) - 1.0)
    cff6 = 0.5 + 0.5 * cp.sign(vbar_stokes[:, 1:dimC]) * vmask_wetMask
    cff7 = 0.5 * vmask_wetMask * cff5 + cff6 * (1.0 - cff5)
    cff1 = cff1 * cff7

    sumArrayV = cp.zeros([dimR, dimC - 1])
    DVSom = vbar_stokes[:, 1:dimC] * cff1
    DVSomSum = cp.sum(DVSom) * kmax

    #     DVomK = cp.zeros([kmax,dimR,dimC-1])
    #     DVom = cp.zeros([kmax,dimR,dimC-1])

    for k in range(kmax):
        DVom = vbarS[k, :, 1:dimC] * cff1
        #         DVomK[k,:,:] = vbarS[k,:,1:dimC]*cff1
        sumArrayV = sumArrayV + DVom

    test1 = cp.sum(sumArrayU)
    test2 = cp.sum(sumArrayV) + DVSomSum
    #     test2 = cp.sum(DVomK) + DVSomSum

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu
print(t_gpu / 1000.0 / iterations)
print(t_cpu / iterations)
print(test1)
print(test2)
# print(DVom.shape)
# print(DVomK.shape)
# print(cff1.shape)