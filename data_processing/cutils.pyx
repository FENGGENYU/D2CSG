import numpy as np
cimport cython


cdef int max_queue_len = 1024*1024*32

@cython.boundscheck(False)
@cython.wraparound(False)
def floodfill(char[:, :, ::1] img, int[:, ::1] queue):
    cdef int dimx,dimy,dimz
    cdef int pi,pj,pk
    cdef int queue_start = 0
    cdef int queue_end = 1

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    
    img[0,0,0] = 0
    queue[queue_start,0] = 0
    queue[queue_start,1] = 0
    queue[queue_start,2] = 0

    while queue_start != queue_end:
        pi = queue[queue_start,0]
        pj = queue[queue_start,1]
        pk = queue[queue_start,2]
        queue_start += 1
        if queue_start==max_queue_len:
            queue_start = 0

        pi = pi+1
        if pi<dimx and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pi = pi-2
        if pi>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pi = pi+1
        pj = pj+1
        if pj<dimy and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pj = pj-2
        if pj>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pj = pj+1
        pk = pk+1
        if pk<dimz and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pk = pk-2
        if pk>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

    for pi in range(dimx):
        for pj in range(dimy):
            for pk in range(dimz):
                if img[pi,pj,pk]>0:
                    img[pi,pj,pk] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
def get_state_ctr(char[:, :, ::1] img, int[:, ::1] state_ctr):
    cdef int dimx,dimy,dimz
    cdef int state,ctr
    cdef int p,i,j,k

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    state = 0
    ctr = 0
    p = 0

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]==state:
                    ctr += 1
                    if ctr==255:
                        state_ctr[p,0] = state
                        state_ctr[p,1] = ctr
                        p += 1
                        ctr = 0
                else:
                    if ctr>0:
                        state_ctr[p,0] = state
                        state_ctr[p,1] = ctr
                        p += 1
                    state = img[i,j,k]
                    ctr = 1

    if ctr > 0:
        state_ctr[p,0] = state
        state_ctr[p,1] = ctr
        p += 1

    state_ctr[p,0] = 2

