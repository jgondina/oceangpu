# !================================================== John C. Warner   ===
# !                                                   Hernan G. Arango   =
# !  Copyright (c) 2002-2010 The ROMS/TOMS Group                         !
# !    Licensed under a MIT/X style license                              !
# !    See License_ROMS.txt                                              !
# !=======================================================================
# !                                                                      !
# !  This routine computes the total heat fluxes.                        !
# !                                                                      !
# !=======================================================================


def atm2ocn_flux(ng):


    # Get local tile arrays
    stfluxT = FORCES[ng].stflux[IstrR:IendR, JstrR:JendR, itemp]
    stfluxS = FORCES[ng].stflux[IstrR:IendR, JstrR:JendR, isal]
    slrflx  = FORCES[ng].srflx [IstrR:IendR, JstrR:JendR]
    lrflx   = FORCES[ng].lrflx [IstrR:IendR, JstrR:JendR]
    shrflx  = FORCES[ng].shflx [IstrR:IendR, JstrR:JendR]
    hrflx   = FORCES[ng].lhflx [IstrR:IendR, JstrR:JendR]
    sustr   = FORCES[ng].sustr [IstrR:IendR, JstrR:JendR]
    evap    = FORCES[ng].evap  [IstrR:IendR, JstrR:JendR]
    rain    = FORCES[ng].rain  [IstrR:IendR, JstrR:JendR]
    rmask   = GRID  [ng].rmask [IstrR:IendR, JstrR:JendR]



    # Atmosphere-Ocean flux computations.
    stfluxT = srflx + lrflx + lhflx + shflx  # fluxes of short and long wave solar radiation + latent and sensible heat.


    # Transfer Taux and Tauy from rho points to U and V points.
    sustr = RtoU(Taux) # Functions to be created
    svstr = RtoV(Tauy)


    # If we include Evaporation minus Precipitation
    if EMINUSP:
        lstfluxS = (1.0/rhow)*(evap - rain)

    # Masks the variables.
    stfluxT[~rmask] = 0
    rain   [~rmask] = 0
    sustr  [~rmask] = 0
    svstr  [~rmask] = 0

    if EMINUSP:
        evap  [~rmask] = 0
        stflux[~rmask] = 0


    if EWperiodic[ng] or NSperiodic[ng]:
    #  Exchange boundary data.

        exchange_r2d_tile(ng, tile, LBi, UBi, LBj, UBj, (stflux[:, :, itemp], rain, sustr, svstr))

        if EMINUSP:
            exchange_r2d_tile(ng, tile, LBi, UBi, LBj, UBj, (evap, stflux[:, :, isalt]))

