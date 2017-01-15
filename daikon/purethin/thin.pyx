# https://codezine.jp/article/detail/98 を参考にしました

import numpy as np
cimport numpy as np

cdef double ne[10]

cdef neighboursC(int x, int y, int** img):
    # image は 0,1 画像データ．　8近傍の０１パターンを状態を左上から反時計回りにならべて返す
    cdef int x_1,y_1,x1,y1
    global ne
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    ne[0],ne[7],ne[6] = img[y_1][x_1], img[y_1][x],  img[y_1][x1]
    ne[1],        ne[5] = img[y][x_1] ,                   img[y][x1]
    ne[2], ne[3],ne[4] =  img[y1][x_1], img[y1][x], img[y1][x1],

cdef bint ifdel(int x, int y, int start, int** img):   # 削除可能かどうかの判定
    cdef int i, k
    cdef double pd, sum
    neighboursC(x, y, img)
    for i in range(3):
        k = start + i
        pd = ne[k % 8]*ne[(k+1) % 8]*ne[(k+2) % 8]
        sum = ne[(k+4) % 8]+ne[(k+5) % 8]+ne[(k+6) % 8]
        if pd == 1 and sum == 0 : 
            return True
    return False

cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)

def getSkelline(img):
    cdef np.ndarray thinnedreturn
    cdef int x,y,i,j,rows,columns,chgnum
    cdef bint change_flag = True
    
    rows, columns = img.shape[0],img.shape[1]    # 画像のサイズ
    
    # thinned = img.copy() / 255
    thinned =  <int**>malloc(sizeof(int*)*rows) # 画像用C配列
    for i in range(rows):
        thinned[i] = <int*>malloc(sizeof(int)*columns)
        for j in range(columns):
            thinned[i][j] = img[i][j]/255
   
    changex = <int*>malloc(sizeof(int)*rows*columns/2) # たかだか画像の面積の半分に収まっていることが前提
    changey = <int*>malloc(sizeof(int)*rows*columns/2)

    while change_flag:
        change_flag = False
        chgnum = 0
        for y in range(1, rows - 1):                   
            for x in range(1, columns - 1): 
                if (thinned[y][x] == 1):
                    if  ifdel(x,y,2,thinned)  :  # UPPER_LEFT
                        changex[chgnum]=x
                        changey[chgnum]=y
                        chgnum = chgnum+1
                        change_flag = True
        for i in range(chgnum):
            thinned[changey[i]][changex[i]] = 0
        
        chgnum = 0        
        for y in range(rows -2 , 0, -1):                   
            for x in range(columns -2, 0, -1): 
                if (thinned[y][x] == 1):
                    if  ifdel(x,y,6,thinned)  :  # LOWER_RIGHT
                        changex[chgnum]=x
                        changey[chgnum]=y
                        chgnum = chgnum+1
                        change_flag = True
        for i in range(chgnum):
            thinned[changey[i]][changex[i]] = 0
        
        chgnum = 0                        
        for y in range(1, rows - 1):                   
            for x in range(columns -2, 0, -1): 
                if (thinned[y][x] == 1):
                    if  ifdel(x,y,0,thinned)  :   # UPPER_RIGHT
                        changex[chgnum]=x
                        changey[chgnum]=y
                        chgnum = chgnum+1
                        change_flag = True
        for i in range(chgnum):
            thinned[changey[i]][changex[i]] = 0
            
        chgnum = 0               
        for y in range(rows -2 , 0, -1):                   
            for x in range(1, columns - 1): 
                if (thinned[y][x] == 1):
                    if  ifdel(x,y,4,thinned) :   # LOWER_LEFT
                        changex[chgnum]=x
                        changey[chgnum]=y
                        chgnum = chgnum+1
                        change_flag = True
        for i in range(chgnum):
            thinned[changey[i]][changex[i]] = 0            
                        
    thinnedreturn = np.zeros((rows,columns),dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            thinnedreturn[i][j] = <double> (thinned[i][j]*255)
        free(thinned[i])
    free(thinned)
    return thinnedreturn
