import numpy as np
cimport numpy as np

cdef double ne[10]

cdef neighboursC(int x, int y, int** img):
    # image は 0,1 画像データ．　8近傍の０１パターンを状態を左隣りから反時計回りにならべて返す
    cdef int x_1,y_1,x1,y1
    global ne
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    ne[0],ne[1],ne[2] = img[y][x_1], img[y1][x_1], img[y1][x]
    ne[3],ne[4],ne[5] =  img[y1][x1], img[y][x1], img[y_1][x1]
    ne[6], ne[7],ne[8] =  img[y_1][x],img[y_1][x_1],img[y][x_1]

cdef neighboursC2(int x, int y, int** img):
    # image は 0,1 画像データ．　8近傍の０１反転パターンを状態を左隣りから反時計回りにならべて返す
    cdef int x_1,y_1,x1,y1
    global ne
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    ne[0],ne[1],ne[2] = 1-img[y][x_1], 1-img[y1][x_1], 1-img[y1][x]
    ne[3],ne[4],ne[5] =  1-img[y1][x1], 1-img[y][x1], 1-img[y_1][x1]
    ne[6], ne[7],ne[8] =  1-img[y_1][x],1-img[y_1][x_1],1-img[y][x_1]
    ne[9] =1- img[y1][x_1]

cdef int transitionsC():
    # 8近傍を反時計回りに1周したときに０から１への変化が何回あるかを返す
    global ne
    cdef int count, i
    count = 0
    for i in range(8):
        count = count + (1 if ne[i]==0 and ne[i+1]==1 else 0)
    return count

cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)

# 連結数 (8連結)
cdef int connection_number():
    global ne
    cdef double count,cn
    cdef int i
    count = 0
    cn = 0
    for i in range(0,7,2):
        cn = cn + ne[i] -  ne[i]*ne[i+1]*ne[i+2]
    for i in range(8):
        count = count + ne[i]   # 反転画像なので８近傍の０の数をカウントしたことになる
    count = 8-count # それを８から引けば１の数をカウントしたことになる
    return int(count*10+cn)  # １０の位が８近傍の１の数で１の位が連結数

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
                    2 <=  s  and  s <= 6   and    
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
                    2 <= s  and  s <= 6  and      
                    transitionsC() == 1 and     
                    P2 * P4 * P8 == 0  and       
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
    # この時点ではまだ８連結と４連結が混ざった線図形なので端点以外の消去可能な画素を削除してしまう
    for y in range(1, rows - 1):
          for x in range(1, columns - 1):
            neighboursC2(x, y, thinned)
            chg1 = connection_number()
            chg2 = chg1/10  # 近傍の１の数を取り出す
            chg1 = chg1 - 10*chg2 # 連結数
            if thinned[y][x]==1 and chg1 == 1 and chg2 > 1:
                thinned[y][x] = 0
                
    free(changex)
    free(changey)
    
    thinnedreturn = np.zeros((rows,columns),dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            thinnedreturn[i][j] = <double> (thinned[i][j]*255)
        free(thinned[i])
    free(thinned)
    return thinnedreturn
    
 
