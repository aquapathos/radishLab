import numpy as np
cimport numpy as np

cdef double ne[9]

cdef neighboursC(int x, int y, int** img):
     # image は 0,1 画像データ．　8近傍の０１パターンを状態を左隣りから時計回りにならべて返す
    cdef int x_1,y_1,x1,y1
    global ne
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    ne[0],ne[1],ne[2] = img[y][x_1], img[y1][x_1], img[y1][x]
    ne[3],ne[4],ne[5] =  img[y1][x1], img[y][x1], img[y_1][x1]
    ne[6], ne[7],ne[8] =  img[y_1][x],img[y_1][x_1],img[y][x_1]
    
cdef int transitionsC():
    # 8近傍を時計回りに1周したときに０から１への変化が何回あるかを返す
    global ne
    cdef int count, i
    count = 0
    for i in range(8):
        count = count + (1 if ne[i]==0 and ne[i+1]==1 else 0)
    return count

cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)

def getSkelline(img):
    cdef np.ndarray thinnedreturn
    cdef int x,y,i,j,rows,columns
    cdef double P2,P3,P4,P5,P6,P7,P8,P9,s
    cdef int ongo, chg1,chg2
    cdef int *imgarr, *changex, *changey
    
    ongo = 1
    rows, columns = img.shape[0],img.shape[1] 
    
    changex = <int*>malloc(sizeof(int)*rows*columns/2) # たかだか画像の面積の半分に収まっていることが前提
    changey = <int*>malloc(sizeof(int)*rows*columns/2)

    # thinned = img.copy() / 255
    thinned =  <int**>malloc(sizeof(int*)*rows) # 画像用C配列
    for i in range(rows):
        thinned[i] = <int*>malloc(sizeof(int)*columns)
        for j in range(columns):
           thinned[i][j] = img[i][j]/255
        
    # changing1 = changing2 = 1        #  the points to be removed (set as 0)
    chg1 = chg2 = 1
    while chg1+chg2  > 0:
        chg1=chg2=0
        # Step 1
        # changing1 = []
        
        for y in range(1, rows - 1):                   
            for x in range(1, columns - 1): 
                neighboursC(x, y, thinned)
                P2,P3,P4,P5,P6,P7,P8,P9 = ne[0],ne[1],ne[2],ne[3],ne[4],ne[5],ne[6],ne[7]
                s = 0
                for i in range(8) :
                    s += ne[i]
                if (thinned[y][x] == 1     and    
                    2 <=  s <= 6   and    
                    transitionsC() == 1 and    
                    P2 * P4 * P6 == 0  and    
                    P4 * P6 * P8 == 0):         
                    # changing1.append((y,x))
                    changex[chg1]=x
                    changey[chg1]=y
                    chg1 = chg1+1
        # for y, x in changing1: 
        #     thinned[y][x] = 0
        for i in range(chg1):
            thinned[changey[i]][changex[i]] = 0
            
        # Step 2
        # changing2 = []
        for y in range(1, rows - 1):
            for x in range(1, columns - 1):
                neighboursC(x, y, thinned)
                P2,P3,P4,P5,P6,P7,P8,P9 = ne[0],ne[1],ne[2],ne[3],ne[4],ne[5],ne[6],ne[7]
                s = 0
                for i in range(8) :
                    s += ne[i]
                if (thinned[y][x] == 1   and       
                    2 <= s <= 6  and      
                    transitionsC() == 1 and     
                    P2 * P4 * P8 == 0 and       
                    P2 * P6 * P8 == 0):            
                    #  changing2.append((y,x))
                    changex[chg2]=x
                    changey[chg2]=y
                    chg2 = chg2+1                                        
        #for y, x in changing2: 
        #    thinned[y][x] = 0
        for i in range(chg2):
            thinned[changey[i]][changex[i]] = 0            
       # print(chg1,chg2)
    free(changex)
    free(changey)
    
    thinnedreturn = np.zeros((rows,columns),dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
           thinnedreturn[i][j] = <double> (thinned[i][j]*255)
        free(thinned[i])
    free(thinned)
    return thinnedreturn
    
 
