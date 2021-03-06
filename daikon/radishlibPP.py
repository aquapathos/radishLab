# 
#  radishibPP.py    2017.09.27
#
import cv2
import matplotlib
import matplotlib.pyplot as plt
# from thin.thin import getSkelline
import numpy as np
import random
from copy import copy
import math

''' ------------------------------------
   cv2 用の画像を matplotlib の imshow で表示するための関数
'''

# 表示をオンオフするためのフラグ
IMGON = True

# プロット用関数

def plotimg(img,figsize=(16,8),layout="111"):
    if IMGON is False: return
    if len(img.shape) == 2:
        pltgry(img,figsize=figsize,layout=layout)
    elif len(img.shape) ==3:
        pltcol(img,figsize=figsize,layout=layout)

def pltgry(img,figsize=(16,8),layout="111"):
    plt.figure(figsize=figsize)
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))

def pltcol(img,figsize=(16,8),layout="111"):
    plt.figure(figsize=figsize)
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
# ２枚の画像をサイズを並べた画像を作って表示する
def paraimage(img1,img2,figsize=(16,8)):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    if img1.ndim == 2:
        img11 = np.zeros((h1,w1,3))
        img11[:,:,0]=img11[:,:,1]=img11[:,:,2]=img1
    else:
        img11=img1
    if img2.ndim == 2:
        img22 = np.zeros((h2,w2,3))
        img22[:,:,0]=img22[:,:,1]=img22[:,:,2]=img2
    else:
        img22=img2
    paraimg = 255*np.ones((max(h1,h2),w1+w2+10,3),dtype=np.uint8)
    
    paraimg[0:h1,0:w1,:] = img11
    paraimg[0:h2,w1+10:,:]=img22
    plotimg(paraimg,figsize=figsize,layout="111")
    
# 余分なマージンをカットする関数
# 余分なマージンをカットした画像を返す関数． オブジェクトサイズ/ratio だけマージンを残す 
# 対象とする画像はシルエットが白のシルエット画像に限定
# ratio :  縁として残す割合　　　デフォルトは４　　　４なら白領域を囲む幅、高さの 1/4 のサイズの縁を残す
# showrec : 囲む矩形を描画するかどうかのフラグ  default は False つまり描かない
# color :  矩形を描く色
# bsize :  矩形の線の太さ

def cutmargin(img,ratio=4, showline = False, color=(255,255,255), bsize = 2) :
    x,y,w,h = cv2.boundingRect(img)
    x2 = int(w/ratio)
    y2 = int(h/ratio)
    shp = (h+2*y2,w+2*x2)
    outimg = np.zeros(shp, dtype=np.uint8) 
    outimg[y2:y2+h,x2:x2+w]=img[y:y+h,x:x+w]
    if showline :
        outimg = cv2.rectangle(outimg,(x2,y2),(x2+w,y2+h),color,bsize)
    return outimg

# 指定された中心周りに指定された角度だけ画像を回転してマージンをカットする関数
# img : 入力画像
# deg : 回転角　degree 
# cx,cy :  回転の中心　　　cx が負の数の場合は画像の中心を基準に回転
# ratio :  縁として残す割合　　

def imagerotation(img, deg,cx= -1, cy=-1 , ratio=4):  # deg 回転角，　(cx,cy) 回転中心  ratio
    if cx < 0:
        cx = img.shape[1]/2
        cy = img.shape[0]/2
    rows,cols = img.shape[:2] 
    need = int(np.sqrt(rows**2+cols**2)) # 描画エリアを十分確保するために対角の長さを計算
    xoff = int((need-cols)/2)
    yoff = int((need-rows)/2)
    tmpimg = np.zeros((need,need),dtype=np.uint8) # 描画用画像エリア
    tmpimg[yoff:yoff+rows,xoff:xoff+cols]=img # シルエット画像をコピー
    mat = cv2.getRotationMatrix2D((cx+xoff,cy+yoff), deg, 1.0)
    outimg = cv2.warpAffine(tmpimg, mat, (0,0))
    outimg = cutmargin(outimg,ratio)
    cv2.threshold(outimg,127,255,cv2.THRESH_BINARY,dst=outimg)
    return outimg

# 回転を考慮した矩形でカットし，それを水平に向けた上でマージンをカット
def rotateAndcut(img,ratio=4) :
    cntimg = copy(img)
    _img,contours,hierarchy = cv2.findContours(cntimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    radishcon = contours[0] # ダイコンの輪郭線
    para_reps = 0.01 # Sufficient accuracy for the radius (distance between the coordinate origin and the line).
    para_aeps = 0.01 # Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.
    # cv2.DIST_L2 は あてはめ誤差の評価基準　https://goo.gl/i0UQag
    [vx,vy,x0,y0] = cv2.fitLine(radishcon, cv2.DIST_L2,0,para_reps,para_aeps)
    deg = math.atan2(vy,vx)*180/np.pi + 180 # 必要な回転角の計算
    outimg = imagerotation(img,deg,x0,y0, ratio)
    return outimg

''' ------------------------------------
   ２階調化と膨張収縮によるゴミ除去
   
   ２階調化した後 
   MORPH_OPEN（収縮後膨張） を２回実行して孤立点や線幅１の髭を除去×２回
   MORPH_CLOSE（膨張後収縮） を２回実行して穴埋め×２回
'''
def dthreshold(image, thres=70, open=True, close=True):
    ret,bw=cv2.threshold(image,thres,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    if open: bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN, kernel,iterations = 2)
    if close: bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE, kernel,iterations = 2)
    return bw

''' ------------------------------------
    ラベルイメージから小領域を削除する
'''
def removeSmallArea(numberoflabels, labelimage, stats, threshold):
    img= np.zeros(labelimage.shape,dtype=np.uint8)
    for i in range(1,numberoflabels ):
        if  stats[i] > threshold:
            img = img + np.uint8(labelimage==i)*255
    return img   

''' ------------------------------------
    白黒画像中の最大面積の連結成分を取り出す
'''
def getBigWhite(image):
    (numberoflabels, labelimage, menseki) = labeling(image,connectivity=4,b_or_w=1)
    # 1番大きな領域がダイコンの主要部分であると仮定する。
    dnum = np.argmax(menseki[1:]) #1番以降を取りだしたときに一番大きい要素を持つ index 
    img = np.uint8(labelimage==dnum+1)*255
    return img

''' ------------------------------------
    白画素の数を答える
'''    
def countWhitePixels(image):
    area = -1
    if image.ndim == 2:
        area = np.sum(image/255)
    return area


''' ------------------------------------
    ダイコン画像と指標画像を分離し、基準となる長さを画素数として求める
    bw ：　画像
    choice : 1  なら1番大きい領域を取り出す。　２なら２番めに大きい領域を取り出す
    output: True ならファイル出力する
    show : True なら表示する
    outfile: 出力ファイル名
    返り値：　ダイコン画像、指標画像、単位長さ辺りのピクセル数、面積
'''    
def getRadish(bw, choice=1, output=False, show = True, outfile=None): 
    if bw.ndim > 2:
        print("Error この画像は白黒画像ではありません")
    cv2.threshold(bw,0,255,cv2.THRESH_BINARY,dst=bw)
    if choice == 1:
        radimg = getBigWhite(bw)  #  もっとも大きな白領域を取り出す。
        stdimg = getBigWhite(bw-radimg)
    if choice == 2 :
        stdimg = getBigWhite(bw)
        radimg = getBigWhite(bw-stdimg)
    (_n, _img, areas) = labeling(stdimg,connectivity=8,b_or_w=1)
    unitlength = np.sqrt(areas[1])
    if show :
        paraimage(radimg,stdimg)
    if output:
        cv2.imwrite(outfile,radimg)
        print("左の画像を{}に出力しました".format(outfile))
    return radimg, stdimg, unitlength, countWhitePixels(radimg)

    
''' ------------------------------------
ラベリングして色つけ
    image  2値画像
    connectivity  8連結型か4連結型かのスイッチ(default 8 )
        4連結の場合は斜めに隣接していても連続していないと考える
    bw 1 なら白領域のラベリング、０なら黒領域のラベリング
    
        関数　
            labeling()  ->  ラベリングしてラベル数、ラベル画像、情報を返す  (gray)
            color2label()  -> ラベル画像に色づけして返す（color)
            colorlabeling()  ->  ラベリングして色付けしてカラー画像を返す（color)
        
    返り値  (numberoflabels, labelimg, menseki) 
         numberoflabels  ラベルの数（反対色全体も０のラベルがつくので１多くなる）  
         labelimg　各画素の明るさデータの代わりにラベルデータが入った画像
         menseki 各ラベルのつけられた画素数のリスト
'''
# image に連結成分ごとのラベルをつけて画像として返す関数
def labeling(image,connectivity=8,b_or_w=1):
    con = labeling2(image,connectivity,b_or_w)
    numberoflabels = con[0]
    labelimg = con[1]
    menseki  = con[2][:,cv2.CC_STAT_AREA]
    return (numberoflabels, labelimg, menseki) 

# labeling2 は connectedcomponentWithStats そのもの。labeling は面積のみ取り出して返す
def labeling2(image,connectivity=8,b_or_w=1):
    img = np.array(image,dtype=np.uint8)
    if b_or_w == 0: img = ~img
    con = cv2.connectedComponentsWithStats(img, connectivity)
    return con 

# ラベル画像に色づけして返す関数
def color2label(labelimage, numberoflabels):
    colors = [ [0,0,0],  # black
                    [255,255,255], # white
                    [255,255,0], # yellow
                    [0,255,255], # aqua
                    [0,255,0], # lime
                    [128,128,0], # olive
                    [255,182,193], # light pink
                    [175,238,238], # paleturquoise
                    [173,255,47], # greeyellow
                    [255,215,0]] # gold
    for  i in range(10,numberoflabels+2):
            colors.append(np.array([random.randint(100, 255), random.randint(0, 255), random.randint(0, 255)]))
    s = labelimage.shape
    colorimage =np.zeros((s[0],s[1],3), dtype=np.uint8)
    
    for y in range(0, s[0]):
       for x in range(0, s[1]):
           if labelimage[y][x] > 0:
                colorimage[y][ x] = colors[labelimage[y][x]]
           else:
                colorimage[y][x] = [0, 0, 0]
    return colorimage

# 上の２つを一気に実行する関数
def colorlabeling(image,connectivity=8,bw=1):
    (numberoflabels, labelimg, menseki) = labeling(image,connectivity,bw)
    colorimage = color2label(labelimg, numberoflabels)
    return (numberoflabels, colorimage, menseki) 

''' ------------------------------------
線図形のセグメンテーション
'''
# 指定されたピクセルの周囲のピクセルを取得
def neighbours(x, y, image):
    ns = [image[y][x+1],image[y-1][x+1],image[y-1][x], image[y-1][x-1],image[y][x-1],  # X１, X２, X３, X４, X5
    image[y+1][x-1],image[y+1][x], image[y+1][x+1],image[y][x+1], image[y-1][x+1] ]    # X6, X7, X8,X1,X2 
    for i in range(10): ns[i] = ns[i]/255
    return ns

# 連結数 (8連結)
def connection_number(neighbours):
    n = 1-np.array(neighbours)
    return sum((1-n)[:-2]), sum( n[i] -  n[i]*n[i+1]*n[i+2]  for  i in range(0,7,2)) 

# スケルトンのリストを生成する関数、細線化バージョン
def skellist(skel, dist):
    # skel 細線化済み画像データ 完全に８連結幅１のひとつながりの線画であることが前提
    # dist 距離画像データ
    # bwの白(255)連結成分を細線化し、残った画素の座標と距離データからなるタプルのリストを返す
    (height,width) = skel.shape
    result = []
 
    for i in range(1,height-1):
        for j in range(1,width-1):
            if skel[i][j]>0: 
                '''
                numberof1, cn = connection_number(neighbours(j,i,skel))
                # この時点でまだ消去可能な画素が残っていれば削除する
                if  skel[i][j] == 255 and  cn==1 and numberof1 > 1  : 
                    skel[i][j] = 0
                else:
                    result.append([i,j,dist[i][j]])'''
                result.append([i,j,dist[i][j]])
    
    return result

''' ------------------------------------
スケルトンからの２値画像データの復元
''' 
# スケルトンデータから2値画像データを復元 （検証のため）
# skeldata は [x,y,d] のリスト、dは距離データ

def skel2img(skeldata,shape):
    img = np.zeros(shape,dtype=np.uint8)
    for i in range (len(skeldata)):
            cv2.circle(img,(np.int(skeldata[i][0]),np.int(skeldata[i][1])),np.int(skeldata[i][2]),(255,255,255),-1)
    for i in range (len(skeldata)):
            cv2.circle(img,(np.int(skeldata[i][0]),np.int(skeldata[i][1])),np.int(3),(0,0,255),-1)
    return img

def skel2img2(skeldata,dimg):
    if len(dimg.shape)==3:
        bgimg = cv2.cvtColor(dimg,cv2.COLOR_BGR2GRAY)
    else:
        bgimg = dimg
    img = cv2.merge((bgimg,bgimg,bgimg))
    for i in range (len(skeldata)):
            cv2.circle(img,(np.int(skeldata[i][0]),np.int(skeldata[i][1])),np.int(skeldata[i][2]),(128,255,0),-1)
    for i in range (len(skeldata)):
            cv2.circle(img,(np.int(skeldata[i][0]),np.int(skeldata[i][1])),np.int(3),(0,0,255),-1)
    return img    
    

# スケルトンと画像を重畳
def skelOnimg(skeldata,bwimg):
    shape = (bwimg.shape[0],bwimg.shape[1],3)
    img = np.zeros(shape,dtype=np.uint8)
    zeroimg = np.zeros(shape,dtype=np.uint8)
    mask = cv2.merge((bwimg,bwimg,bwimg))
    for i in range (len(skeldata)):
        cv2.circle(img,(int(skeldata[i][0]),int(skeldata[i][1])),3,(255,255,255),-1)
#    plt.imshow(mask)
    return cv2.bitwise_xor(img,mask)  

'''' ------------------------------------
スケルトンを分岐点で分割

完全に幅１の８連結線画であることが前提
'''
def skllabel(skelimg,  skeldata):
    con = True
    tmpimg = np.array(skelimg,dtype=np.uint8)
    count = 0
    while con:
        con = False
        for skelton in skeldata:
            x = skelton[1]
            y = skelton[0]
            numberof1, cn = connection_number(neighbours(x,y,tmpimg))
            if tmpimg[y][x] == 255 and cn > 2 :
                # 線画から分岐点の画素を除去する
                con = True
                count += 1
                tmpimg[y][x] = 0
    if count == 0:
        out = (2,skelimg,[-1,-1])
    else:
        out = labeling(tmpimg,connectivity=8,b_or_w=1)
    return  out #(ラベル数，　ラベル画像，　ラベル諸表のリストが返り値）

''' ------------------------------------
スケルトンの髭を除去する
'''
def oneSkelton(skelimg, skeldata):
    # まず分岐点で線画を分割してラベル画像を得る
    (ns, labelimage, specs) = skllabel(skelimg,  skeldata)
    if ns == 2:   # ラベル数が２のときは何もしなくていい
        pass
    else:
        con = True
        while con:
            tmpimg = np.array(skelimg,dtype=np.uint8)  # 元のイメージをコピー
            shortestsegment = np.argmin(specs)  # 一番短いセグメント番号
            for y in range(0,skelimg.shape[0]):
                for x in range(0,skelimg.shape[1]):
                    if labelimage[y][x] == shortestsegment:
                        tmpimg[y][x] = 0  # 画素を削除する
            check = labeling(tmpimg,connectivity=8,b_or_w=1)
            numberofregions = check[0]
            if numberofregions > 2:  # もともと２なのに増えたということは中間のセグメントであった
                specs = specs[:shortestsegment]+specs[shortestsegment+1:]
            else:
                skelimg = tmpimg
                sknew = []
                for d in skeldata:
                    if  skelimg[d[0]][d[1]] > 0 :
                        sknew.append(d)
                skeldata = sknew
                (ns, labelimage, specs) = skllabel(skelimg,  skeldata) 
                if ns == 2:
                    con = False
    return skelimg, skeldata

# スケルトンの位置データ補正
def recalcDistanceP(skdata,normP):
    skdpbackup=np.array(skdata)
    # 曲線に沿った距離を求める。理屈では積分していけばいいが
    # デジタル画像では近い画素間の距離は誤差が大きいので10点ごとに
    # 基準点を設け、基準点間の距離の積算＋最寄りの基準点からの距離を
    # 曲線に沿った距離の近似に使う
    xnorm = skdata[normP][1]
    ynorm = skdata[normP][0]
    cnt = 0 # ベースからの画素数 
    accdistAtBase = 0 # 基準点から現在のベースまでの距離の積算
    predist = skdpbackup[0][3]  # 先端のデータ（最も誤差が大きいはず）
    for sk in skdata[normP-1::-1]:
            y = sk[0]
            x = sk[1]
            sk[3] = accdistAtBase - np.sqrt((x-xnorm)**2 + (y - ynorm)**2) 
            cnt = cnt+1
            if cnt==10 :
                accdistAtBase = sk[3]
                (ynorm,xnorm) = (y,x)
                cnt = 0
    print(u"左側修正量 {} ({} -> {})".format(predist-skdata[0][3],predist,skdata[0][3]))
    xnorm = skdata[normP][1]
    ynorm = skdata[normP][0]
    cnt = 0 # ベースからの画素数 
    accdistAtBase = 0 # 基準点から現在のベースまでの距離の積算
    predist = skdpbackup[-1][3]  # 葉元のデータ（こちらも誤差が大きいはず）
    for sk in skdata[normP+1::]:
            y = sk[0]
            x = sk[1]
            sk[3] = accdistAtBase + np.sqrt((x-xnorm)**2 + (y - ynorm)**2) 
            cnt = cnt+1
            if cnt==10 :
                accdistAtBase = sk[3]
                (ynorm,xnorm) = (y,x)
                cnt = 0
    print(u"右側修正量 {} ({} -> {})".format(skdata[-1][3]-predist,predist,skdata[-1][3]))


'''
描画関数

radiusfunc(invert)
軸に沿った距離を横軸，　その点の距離データを縦軸にしたグラフを描く
（距離データとは軸上の点から表皮までの最短距離．　　中央辺りでは径に相当する

'''
def radiusfunc(skdp,invert=True,):
    if invert:
        for d in skdp: d[3]=-d[3]
    skdN = skdp[skdp[:,3]<= 0] # 基準点より左のデータ
    skdP = skdp[skdp[:,3] > 0] # 基準点より右のデータ
    
    # 横軸を基点からのスケルトン位置までの距離，縦軸をそのスケルトンの距離データとしたグラフ                          
    plt.axis('equal')    
    plt.hold(True)
    ydata = np.array(skdN[:,2])
    xdata = np.array(skdN[:,3])
    plt.plot(xdata,ydata,'-',color=(1,0,0.0))
    plt.plot(xdata,-ydata,'-',color=(1,0,0.0))
    ydata = np.array(skdP[:,2])
    xdata = np.array(skdP[:,3])
    plt.plot(xdata,ydata,'-',color=(0,0,1.0))
    plt.plot(xdata,-ydata,'-',color=(0,0,1.0))
    plt.hold(False)
    plt.show()
    
    
# スケルトンを直線状に再配置し、曲がりを補正した画像を生成する
def makeNormalizedImage(skdp):
    print("中心軸の画素数={}".format(len(skdp)))
    maxh = np.max(skdp[:,2]) # もっとも近い輪郭までの距離
    maxw = np.max(skdp[:,3]) # 基準点からの距離の最大値
    minw = np.min(skdp[:,3]) # 基準点からの距離の最小値
    nimg = np.zeros((int(3*maxh),int(1.5*(maxw-minw)),3),dtype=np.uint8) # 50%マージンで描画用エリアを確保
    cent = np.array((int(1.5*(-minw)),int(1.5*maxh))) # 基準点の画像内座標
    for i in range(len(skdp)):
        center = (int(skdp[i][3])+cent[0],cent[1])
        radius = int(skdp[i][2])
        color = (255,255,255) # White
        thickness = -1 # 塗りつぶし
        nimg = cv2.circle(nimg,center,radius,color,thickness)
    mask = nimg[:,cent[0]:,:]
    cv2.threshold(mask,127,128,cv2.THRESH_BINARY,dst=mask)
    nimg = cv2.cvtColor(nimg,cv2.COLOR_BGR2GRAY)
    return nimg

# 輪郭線の抽出
# mode 0  全体の輪郭　　mode 1 最大径より先だけの輪郭  mode 2 先の上半分だけ
def  getContour(img, mode):
    center = img.shape[0]/2
    if mode == 0 or mode == 3:
        ret, bw = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
    else:
        ret, bw = cv2.threshold(img,200,255,cv2.THRESH_BINARY)  
    image, contours, hierarchy = cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if  mode ==2:
        cnt = []
        maxx = np.max(contours[:][0][1])
        for i in contours[0]:
            if i[0][1] >= center and i[0][0]<maxx :
                cnt.append([list(i[0])])
            contours = np.array([cnt])
    if  mode ==3:
        cnt = []
        maxx = np.max(contours[:][0][1])
        for i in contours[0]:
            if i[0][1] >= center:
                cnt.append([list(i[0])])
            contours = np.array([cnt])   
    return image,contours


# 先端部分だけの点列の生成
def getTipData(image):
    bw,cnt = getContour(image,mode=2)
    xdata = [i[0][0]  for i in cnt[0]]
    ydata = [i[0][1]- bw.shape[0]/2 for i in cnt[0]] 
    return xdata,ydata

def getTipDataAll(image):
    bw,cnt = getContour(image,mode=3)
    xdata = [i[0][0]  for i in cnt[0]]
    ydata = [i[0][1]- bw.shape[0]/2 for i in cnt[0]] 
    return xdata,ydata

# yの値が target となる位置が x=0 となるようにデータ全体を x 方向にシフトさせる
def shiftX(dx,dy,target):
         found = False
         offset = 0.0
         for (x,y) in zip(dx,dy):
            if not found and y >target:
                found = True
                offset = x
         dx = dx - offset
         return dx,dy

#  norm を基準としてサイズを正規化 
def sizeNormalize(xdata,ydata, norm, target):
    xdataR = np.array(xdata)/norm 
    ydataR= np.array(ydata)/norm
    xdataR= shiftX(xdataR,ydataR,target)    
    return xdataR, ydataR


# 画像にベクトルを描画
def drawAxis(img, start, vec, colour, length):
    # 終了点
    end = (int(start[0] + length * vec[0]), int(start[1] + length * vec[1]))

    # 中心を描画
    cv2.circle(img, (int(start[0]), int(start[1])), 5, colour, 2)

    # 軸線を描画
    cv2.line(img, (int(start[0]), int(start[1])), end, colour, 2,cv2.LINE_AA) # LINE_AA アンチエイジング
    

    # 先端の矢印を描画
    angle = math.atan2(vec[1],vec[0])

    qx0 = int(end[0] - 9 * math.cos(angle + math.pi / 4))
    qy0 = int(end[1] - 9 * math.sin(angle + math.pi / 4))
    cv2.line(img, end, (qx0, qy0), colour, 1, cv2.LINE_AA)   

    qx1 = int(end[0] - 9 * math.cos(angle - math.pi / 4))
    qy1 = int(end[1] - 9 * math.sin(angle - math.pi / 4))
    cv2.line(img, end, (qx1, qy1), colour, 1, cv2.LINE_AA)

# rimg 画像
# points cv2.findContours が返す輪郭線データ
# dotsize 輪郭線を描く線の太さ
# ldotsize 飛び飛びに塗る点の点の太さ
# sdotsize 始点の点の太さ
# interval 飛び飛びに異なる色で塗る点の間の間隔
# flags 中間点や始点を別色で目立つように表示するかどうか
def drawRadishContour(rimg, points, dotsize =1,ldotsize=3, sdotsize=10,interval=20,flag=False):
    sh = rimg.shape
    zeroimg = np.zeros(sh,dtype=np.uint8)
    pltimg = np.zeros((sh[0],sh[1],3),dtype=np.uint8)
    pltimg = cv2.merge((rimg,rimg,rimg))
    for i in range (len(points)):
            cv2.circle(pltimg,(points[i][0],points[i][1]),dotsize,(0,255,255),-1)
    if flag :
        for i in range (0,len(points),interval):
                cv2.circle(pltimg,(points[i][0],points[i][1]),ldotsize,(255,0,0),-1)
        cv2.circle(pltimg,(points[0][0],points[0][1]),sdotsize,(0,0,255),-1)
    return pltimg
    
# 各連続３点それぞれについて３つのベクトルを求める
# pts サンプル点
# samplefactor  この数でサンプル数を割った値だけ離れた3点で曲率を求める
# masklen ミディアンフィルタで雑音除去する際に、前後この値まで離れた点を使う
# open  両端が開いている場合True、　閉路の場合　False
#     スケルトンの曲率を求める場合は　　True , 輪郭の場合は False
def curvature(pts, samplefactor = 200,masklen=20, open=False):
    
    sd = int(len(pts)/samplefactor)  # 曲率計算に用いるサンプル３点の間隔
    print("曲率計算のためのサンプル間距離は",sd)
    
    # ミディアンフィルタのサイズが指定されていない場合は sd をサイズとして用いる
    if masklen < 0:  
        masklen = sd
    print("ミディアンフィルタのサイズ", masklen)

    v1 = np.zeros_like(pts)
    v2 = np.zeros_like(pts)
    v3 =  np.zeros_like(pts)
    v1[0:sd] = pts[-sd:] - pts[0:sd]  # P[i-1] - P[i]   （のリスト　　以下省略）
    v1[sd:] = pts[0:-sd] - pts[sd:]  
    v2[0:-sd] = pts[sd:] - pts[0:-sd]   # P[i +1]- P[i]  
    v2[-sd:] = pts[0:sd] - pts[-sd:]  
    v3[0:sd] = pts[sd:2*sd] - pts[-sd:] # p[i+1]-P[i-1]
    v3[sd:-sd] = pts[2*sd:] - pts[0:-2*sd]
    v3[-sd:]=pts[0:sd] - pts[-2*sd:-sd]
    
    # ベクトルのノルム（長さ）を求める
    v1s = np.array([np.sqrt(np.dot(v1[i],v1[i]))  for i in range(len(pts))]) #  V1 のノルム（長さ）
    v2s = np.array([np.sqrt(np.dot(v2[i],v2[i]))  for i in range(len(pts))]) # V２ のノルム（長さ）
    v3s = np.array([np.sqrt(np.dot(v3[i],v3[i]))  for i in range(len(pts))]) # V３ のノルム（長さ）　　

    ppp = v1s*v2s*v3s # ３辺の長さの積（のリスト）

    cp = np.zeros(len(pts)) # 外積のノルムの大きさ格納用配列
    cp = -np.cross(v2, v1)

    # 曲率を求める
    cuv = np.zeros(len(pts)) 
    cuv = 2*cp/ppp
    
    # デジタルノイズの除去のためにミディアンフィルタをかける
    if masklen > 0:
        cvm =  copy(np.r_[cuv[-masklen:],cuv,cuv[0:masklen]])
        for  i in range(len(cuv)):
            cuv[i] = np.median(cvm[i:i+2*masklen]) 
     
    # 始点から曲線に沿って測った距離を各点について求める。
    # samplefactor で指定した数のサンプル点をマイルストーンとし、マイルストーン間は曲率から距離を求める
    
    sn = int(len(pts)/sd)  
    if (len(pts)%sd) == 0:
         samplesNeeded = sn
    else:
         samplesNeeded = sn+1
    ｍｓ = np.zeros(samplesNeeded) 
    ｍｓ[0]=0
    for idx in range(samplesNeeded-1) :
        i = idx+1
        k = cuv[i*sd]
        kr = 1/k if k>0 else np.float('inf') # サンプル点の曲率半径
        sv = pts[i*sd]-pts[(i-1)*sd] # サンプル間を結ぶベクトル
        s = np.linalg.norm(sv)
        ms[i] = ms[i-1]+(2*math.asin(s/2.0/kr)*kr if kr < np.float('inf') else s)

    #  print("曲率 {:.4f}  前の点からの距離 {:.4f}   累積距離 {}".format(k,s,ms[i]))
    dl = np.zeros(len(pts))  # 折れ線にそって測った距離の格納場所を確保
    for i in range(len(pts)):
        if i % sd == 0:
            s = ms[int(i/sd)]
        else:
            s += np.linalg.norm(pts[i]-pts[i-1])
        dl[i] = s;
        
    if open :
        cuv=cuv[sd:-sd]
        dl = dl[sd:-sd]
    cvmaxind = np.argmax(cuv)
    cvminind = np.argmin(cuv)
    if open:
        tip = pts[cvmaxind+sd] 
        tipmin = pts[cvminind+sd] 
    else:
        tip = pts[cvmaxind] 
        tipmin = pts[cvminind]
    print("曲率最大の点は",cvmaxind,"番のサンプル点で、曲率は",  cuv[cvmaxind],"座標は", tip)
    print("曲率最小の点は",cvminind,"番のサンプル点で、曲率は",  cuv[cvminind],"座標は", tipmin)
    
    return dl , cuv, tip  
    # pts: サンプル点,  
    # dl : サンプルの始点から測った折れ線近似された輪郭に沿って各点までの距離
    # cuv: 曲率のリスト

'''
山登りによるスケルトン抽出

radiusfunc(invert)
軸に沿った距離を横軸，　その点の距離データを縦軸にしたグラフを描く
（距離データとは軸上の点から表皮までの最短距離．　　中央辺りでは径に相当する

'''   
# スケルトン追跡
# 座標(x,y) の８近傍で最も黒画素までの距離の大きな点を求める

def get8nb(x,y,im):
     # (x,y)の画素を中心に、右隣から時計回りに8近傍データを取得
     netmp = np.array([im[y,x+1],im[y+1,x+1],im[y+1,x],im[y+1,x-1],im[y,x-1], im[y-1,x-1],im[y-1,x],im[y-1,x+1]])
     return netmp

# 最大曲率点からのスケルトントレースで次の点を決定する関数
# x,y 現在地点、次の地点を決める基準方向、sx,sy 最大曲率点、para1 斜め方向の割引率
def getNextRidge(x,y,direct,im,cim, sx,sy, para1 = 0.998):
    if im[y,x] == 0:
        print("Out of object area")
        return None,None,None
    # (x,y) の次の点を direct 方向とその左右のうちから選ぶ。
    # direcct は方向
    # im は作業用距離データ
    # cimは(x,y)の距離データ
    R2 = np.sqrt(2.0)/2.0
    VEC =  np.array([[1,0],[R2,R2],[0,1],[-R2,R2],[-1,0],[-R2,-R2],[0,-1],[R2,-R2]])
    nx = x + [1,1,0,-1,-1,-1,0,1][direct]
    ny = y + [0,1,1,1,0,-1,-1,-1][direct]
    nRx = x + [1,1,0,-1,-1,-1,0,1][(direct+1)%8]
    nRy= y + [0,1,1,1,0,-1,-1,-1][(direct+1)%8]
    nLx = x + [1,1,0,-1,-1,-1,0,1][(direct-1)%8]
    nLy= y + [0,1,1,1,0,-1,-1,-1][(direct-1)%8]
    nb3=np.array( [im[nLy,nLx], im[ny,nx],im[nRy,nRx]])
    # 斜め方向は移動距離が違い、上下左右より若干有利になるので少し割り引いて評価
    if direct % 2 == 0:
        if nb3[0] > 0:
            nb3[0] = cim + para1*(nb3[0]-cim) 
        if nb3[2] > 0:    
            nb3[2] = cim + para1*(nb3[2]-cim) 
    else:
        if nb3[1] > 0:
             nb3[1] = cim + para1*(nb3[1]-cim)   
                
    # 初期位置に近づく方向は除外する。また、目標方向とその左右が同点なら目標方向を優先する。
    
    distance0 = np.linalg.norm([sx-x,sy-y])
    distanceN = np.linalg.norm([sx-nx,sy-ny])
    distanceR = np.linalg.norm([sx-nRx,sy-nRy])
    distanceL = np.linalg.norm([sx-nLx,sy-nLy])
   
    if distanceN <= distance0:
        nb3[1] = 0
    if distanceR <= distance0  or distanceR == distanceN or nb3[0] == nb3[1]:
        nb3[2] = 0
    if distanceL <= distance0  or distanceL == distanceN or nb3[2] == nb3[1]:
        nb3[0] = 0
       
    # print("{:4.4f} {:4.4f}  {:4.4f}  {:4.4f} ".format(distance0,distanceN-distance0,distanceR-distance0,distanceL-distance0))
    if np.max(nb3) > 0:        
        idx = np.argmax(nb3) # 3方向の中で最も大きな値のインデックス
        nx = x + [1,1,0,-1,-1,-1,0,1][(direct+idx-1)%8]
        ny = y + [0,1,1,1,0,-1,-1,-1][(direct+idx-1)%8]
        # print("idx",idx,"next","direct",direct, "direct+idex-1",direct+idx-1,nx,ny)
        return (direct+idx-1)%8, nx, ny
    else :
        nb = get8nb(x,y,im)
        idx = np.argmax(nb) # 3方向がすべて０の場合は８近傍で最も大きな値のインデックス
        nx = x + [1,1,0,-1,-1,-1,0,1][idx]
        ny = y + [0,1,1,1,0,-1,-1,-1][idx]
        print("No way!")
        return idx, nx, ny

# ******* 最大曲率点からのスケルトントレース
# sx,sy スタート地点、direct 初期追跡方向、
# phaseswitch この太さの地点を通り過ぎるまでは周辺が黒画素から距離１でも先に進む
# para1 斜め方向の割引率
def traceRidges1(sx,sy,direct,img, phaseswitch=5, para1 = 0.998):
    USED = -99999
    cim = img[sy,sx]
    skel=[[sx,sy,cim]]
    img[sy,sx] = USED
    mv = 1
    phase = 0
    x = sx
    y = sy
    while  (phase == 0 and mv >  0.4 )  or ( phase > 0 and mv > 1) :
        if phase == 0 and mv > phaseswitch : 
            phase = 1  # トレースがある程度太い部分まで至った
        direct, x,y = getNextRidge(x,y,direct,img,cim,sx,sy, para1=0.998)
        mv = img[y,x]
        if mv > 0:
            skel.append([x,y,mv])
            img[y,x] = USED
            mx1 = x + [1,1,0,-1,-1,-1,0,1][(direct+5)%8]
            my1 = y + [0,1,1,1,0,-1,-1,-1][(direct+5)%8]
            mx2 = x + [1,1,0,-1,-1,-1,0,1][(direct+3)%8]
            my2 = y + [0,1,1,1,0,-1,-1,-1][(direct+3)%8]  
            img[my1,mx1] = USED
            img[my2,mx2] = USED
    return skel    
    
##  ----   このメソッドは古い。両端点からのトレースには下の traceRidgePP に置き換わりました。
##  古いプログラムを動かすために残してあります。
def getNextRidge2(x,y,direct, im ,cim, mx,my, para1, para2):
    # (x,y) の次の点を direct 方向とその左右のうちから選ぶ。
    # direcct は方向
    # im は作業用距離データ
    # cimは(x,y)の距離データ
    R2 = np.sqrt(2.0)/2.0
    VEC =  np.array([[1,0],[R2,R2],[0,1],[-R2,R2],[-1,0],[-R2,-R2],[0,-1],[R2,-R2]])
    nx = x + [1,1,0,-1,-1,-1,0,1][direct]
    ny = y + [0,1,1,1,0,-1,-1,-1][direct]
    nRx = x + [1,1,0,-1,-1,-1,0,1][(direct+1)%8]
    nRy= y + [0,1,1,1,0,-1,-1,-1][(direct+1)%8]
    nLx = x + [1,1,0,-1,-1,-1,0,1][(direct-1)%8]
    nLy= y + [0,1,1,1,0,-1,-1,-1][(direct-1)%8]
    nb3=np.array( [im[nLy,nLx], im[ny,nx],im[nRy,nRx]])
    # 斜め方向は移動距離が違い、上下左右より若干有利になるので少し割り引いて評価
    if direct % 2 == 0:
        if nb3[0] > 0:
            nb3[0] = cim + para1*(nb3[0]-cim) 
        if nb3[2] > 0:    
            nb3[2] = cim + para1*(nb3[2]-cim) 
    else:
        if nb3[1] > 0:
             nb3[1] = cim + para1*(nb3[1]-cim) 
    # 目標から遠ざかる方向は除外する。また、目標方向とその左右が同点なら目標方向を優先する。
    vecd = np.linalg.norm([mx-x,my-y])
    if np.dot(VEC[(direct-1)%8],[mx-x,my-y])/vecd  < para2   or nb3[0] == nb3[1]:
         nb3[0] = 0
    if np.dot(VEC[(direct+1)%8],[mx-x,my-y])/vecd  < para2  or nb3[0] == nb3[2]:
         nb3[2] = 0
    if np.max(nb3) > 0:
         idx = np.argmax(nb3) # 3方向の中で最も大きな値のインデックス
    else: # ３方向すべて０かすでに訪れた場所である場合
        idx = 1
    if idx == 0:
        nx = nLx
        ny = nLy
    elif idx == 2:
        nx = nRx
        ny = nRy
    return  nx, ny



# 上下左右斜め8方向のうち、目標に向う方向に近い方向番号を求める
# 方向番号は右を０として、時計回り
#        5  6  7
#        4     0
#        3  2  1
def decideDirection(x,y,mx,my):
    R2 = np.sqrt(2)/2.0
    vec = np.array([[1,0],[R2,R2],[0,1],[-R2,R2],[-1,0],[-R2,-R2],[0,-1],[R2,-R2]])
    dx = mx - x
    dy = my - y
    dotp = np.dot(vec, np.array([dx,dy]))  # 内積のリスト
    return np.argmax(dotp)

# *******
# 目標(maxX, maxY) 向きの３方向のうちで最大傾斜方向へ、目標に至るまで進む
# direct = 0  右向きにサーチ
# direct = 4  左向きにサーチ
# para1 斜め移動の割引率　 0.8〜1.0  default 0.998
# para2 角度の偏移の排除基準  cos の値で指定　　０~１　　default 0.25
def traceRidges2(x,y,dist, mx,my, para1= 0.998,para2 = 0.25):
    USED = -99999
    cim = dist[y,x]
    skel = [[x,y,dist[y,x]]]
    dist[y,x] = USED
    while  (x - mx)*(x - mx)+(y - my)*(y - my) > 2:       
        direct = decideDirection(x,y,mx,my)
        x,y = getNextRidge2(x,y,direct,dist,cim,mx,my, para1=para1,para2=para2)
        skel.append([x,y,dist[y,x]])
        # print(u"{}  (x,y) {:4d} {:4d}, d {:7.3f}, goal {:4d}　{:4d}, dif {:3d} {:3d}".format(direct,x,y,dist[y,x],mx,my,mx-x,my-y))
        dist[y,x] = USED    

    return skel


def getNP(x,y,direct):    # 点 (x,y) から direct 方向　を中心とした５点の座標を返すメソッド
    dvecX = [1,1,0,-1,-1,-1,0,1]
    dvecY = [0,1,1,1,0,-1,-1,-1]
    nx = x + dvecX[direct]
    ny = y + dvecY[direct] 
    nRx =  x + dvecX[(direct+1)%8]
    nRy =  y + dvecY[(direct+1)%8]
    nLx = x + dvecX[(direct-1)%8]
    nLy = y + dvecY[(direct-1)%8]
    nRx2 =  x + dvecX[(direct+2)%8]
    nRy2 =  y + dvecY[(direct+2)%8]
    nLx2 = x + dvecX[(direct-2)%8]
    nLy2 = y + dvecY[(direct-2)%8]
    return [(nLy2,nLx2),(nLy,nLx),(ny,nx),(nRy,nRx),(nRy2,nRx2)]

# ----   ここまでの新しいメソッドが下にある traceRidgePP 

######################
#   traceRidgePP
######################
def getNextRidgePP(dist, x, y, direct, mx,my, para1=0.7071):
    # dist 距離画像データ
    # (x,y) の次の点を direct 方向とその左2,右2 ５方向のうちから選ぶ。
    # direct は探索方向 
    # (mx,my) 最終到達目標点
    
    ppac5 = getNP(x,y,direct)  # direct 方向を中心とした近傍5点を得る
    [(nLy2,nLx2),(nLy,nLx),(ny,nx),(nRy,nRx),(nRy2,nRx2)]=ppac5 # 5近傍それぞれの座標

    cim = dist[y,x]  # 点(x,y) の距離データ
    dnb2cp5 = np.array([dist[i,j] for i,j in ppac5])-cim # 5近傍の距離データとcimの差分

    # 斜め方向は移動距離が違い、上下左右より若干有利になるので少し割り引いて評価
    if direct % 2 == 0:
            dnb2cp5[1] *= para1
            dnb2cp5[3] *= para1
    else:
            dnb2cp5[0] *= para1
            dnb2cp5[2] *= para1
            dnb2cp5[4] *= para1             
    dmaxes = np.where(dnb2cp5==dnb2cp5.max())  # ５方向の中でもっとも勾配の大きい方向
    if dmaxes[0].size>1:
        idx = int(np.average(dmaxes[0]))
    elif dmaxes[0].size == 1:
        idx = np.argmax(dnb2cp5) # 5方向の中で最も大きな値のインデックス
        if  idx != dmaxes[0]:
            print('not same', idx, dmaxes[0])
    else: # 5方向すべて０かすでに訪れた場所である場合は探索方向を維持
        idx = 2
    
    #print(x,y," {:3.2f}, {:3.2f}, {:3.2f}, {:3.2f}, {:3.2f} - >{}".format(dnb2cp5[0],dnb2cp5[1],dnb2cp5[2],dnb2cp5[3],dnb2cp5[4],idx))

    (ny,nx) = ppac5[idx]
    newdirect = (direct + (idx-2))%8
    return  nx, ny, newdirect


def traceRidgePP(dist,x,y, mx,my, para1= 0.7071):
    USED = -99999
    cim = dist[y,x] # 現在地の距離情報（書き換わる前）
    skel = [[x,y,dist[y,x]]]
    direct = decideDirection(x,y,mx,my)
    
    count = 0
    while  (x - mx)*(x - mx)+(y - my)*(y - my) > 2 and count < 5000:
        count +=1
        nx,ny,nd = getNextRidgePP(dist, x,y,direct, mx,my, para1=para1)
        skel.append([nx,ny,dist[y,x]])
        dist[y,x] = USED
        x,y,direct = nx,ny,nd  
    return skel

##########


# スケルトンの位置データ補正
def recalcDistanceP(skdata,normP):
    skdpbackup=np.array(skdata)
    # 曲線に沿った距離を求める。理屈では積分していけばいいが
    # デジタル画像では近い画素間の距離は誤差が大きいので10点ごとに
    # 基準点を設け、基準点間の距離の積算＋最寄りの基準点からの距離を
    # 曲線に沿った距離の近似に使う
    xnorm = skdata[normP][0]
    ynorm = skdata[normP][1]
    cnt = 0 # ベースからの画素数 
    accdistAtBase = 0 # 基準点から現在のベースまでの距離の積算
    predist = skdpbackup[0][3]  # 先端のデータ（最も誤差が大きいはず）
    for sk in skdata[normP-1::-1]:
            y = sk[1]
            x = sk[0]
            sk[3] = accdistAtBase - np.sqrt((x-xnorm)**2 + (y - ynorm)**2) 
            cnt = cnt+1
            if cnt==10 :
                accdistAtBase = sk[3]
                (ynorm,xnorm) = (y,x)
                cnt = 0
    print(u"左側修正量 {} ({} -> {})".format(predist-skdata[0][3],predist,skdata[0][3]))
    xnorm = skdata[normP][0]
    ynorm = skdata[normP][1]
    cnt = 0 # ベースからの画素数 
    accdistAtBase = 0 # 基準点から現在のベースまでの距離の積算
    predist = skdpbackup[-1][3]  # 葉元のデータ（こちらも誤差が大きいはず）
    for sk in skdata[normP+1::]:
            y = sk[1]
            x = sk[0]
            sk[3] = accdistAtBase + np.sqrt((x-xnorm)**2 + (y - ynorm)**2) 
            cnt = cnt+1
            if cnt==10 :
                accdistAtBase = sk[3]
                (ynorm,xnorm) = (y,x)
                cnt = 0
    print(u"右側修正量 {} ({} -> {})".format(skdata[-1][3]-predist,predist,skdata[-1][3]))
    
   
# スケルトンを直線状に再配置し、曲がりを補正した画像を生成する
def makeNormalizedImage(skdp):
    print("中心軸の画素数={}".format(len(skdp)))
    maxh = np.max(skdp[:,2]) # もっとも近い輪郭までの距離
    maxw = np.max(skdp[:,3]) # 基準点からの距離の最大値
    minw = np.min(skdp[:,3]) # 基準点からの距離の最小値
    nimg = np.zeros((int(3*maxh),int(1.5*(maxw-minw)),3),dtype=np.uint8) # 50%マージンで描画用エリアを確保
    cent = np.array((int(1.5*(-minw)),int(1.5*maxh))) # 基準点の画像内座標
    for i in range(len(skdp)):
        center = (int(skdp[i][3])+cent[0],cent[1])
        radius = int(skdp[i][2])
        color = (255,255,255) # White
        thickness = -1 # 塗りつぶし
        nimg = cv2.circle(nimg,center,radius,color,thickness)
    mask = nimg[:,cent[0]:,:]
    cv2.threshold(mask,127,128,cv2.THRESH_BINARY,dst=mask)
    nimg = cv2.cvtColor(nimg,cv2.COLOR_BGR2GRAY)
    return nimg

# ファイルオープンダイアログ
input_form = """
<div style="border:solid navy; padding:20px;">
<input type="file" id="file_selector" name="files[]"/>
<output id="list"></output>
</div>
"""

javascript = """
<script type="text/Javascript">
  function handleFileSelect(evt) {
    var kernel = IPython.notebook.kernel;
    var files = evt.target.files; // FileList object
    console.log('Executing orig')
    console.log(files)
    // files is a FileList of File objects. List some properties.
    var output = [];
    var f = files[0]
    output.push('<li><strong>', escape(f.name), '</strong> (', f.type || 'n/a', ') - ',
                  f.size, ' bytes, last modified: ',
                  f.lastModifiedDate ? f.lastModifiedDate.toLocaleDateString() : 'n/a',
                  '</_Mli>');
    document.getElementById('list').innerHTML = '<ul>' + output.join('') + '</ul>';
    var command = 'fname = "' + f.name + '"'
    console.log(command)
    kernel.execute(command);
  }

  document.getElementById('file_selector').addEventListener('change', handleFileSelect, false);
</script>
"""

def file_selector():
    from IPython.display import HTML, display
    display(HTML(input_form + javascript))
    