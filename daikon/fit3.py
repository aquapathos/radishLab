# TensorFlow による曲線あてはめ   pattern 3
import matplotlib
# %matplotlib nbagg
# matplotlib.use('nbagg') 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#  y = 1/（1+exp(-a (x-c)(1- exp(-b(x-c)) ))という関係式を使って
# 実験用に仮想の x, y のペアデータを生成し、
# TensorFlow で回帰曲線を求めてみたい。

    
def fit3(data,itr=1000,alpha=1.0,xmin=-5,xmax=5,cutoff=0.2):  # itr 繰り返しの回数、alpha 学習係数、cutoff y の値がこれ以下の部分は近似に使わない


    def model(a,b,c,x):  # 1/(1+exp(-a*(x-c)*(1+ exp(-b*(x-c)))))
                x_c = tf.subtract(x,c) # x - c
                tmpB = tf.exp(-tf.multiply(b,x_c))   # exp( - b ( x-c) )
                tmpB1 = tf.add(1.0,  tmpB )   # 1 + exp(- b ( x-c) )
                tmpAB  = tf.exp(-tf.multiply(tf.multiply(a,x_c),tmpB1))  # exp( - a ( x-c)*(1+exp(b-c) )
                ret = tf.div(1.0,tf.add(1.0,tmpAB))
                return ret

    (x_dataAll,y_dataAll) = data
    use = y_dataAll >= cutoff
    x_data=x_dataAll[use]
    y_data=y_dataAll[use]
    x_dataRest = x_dataAll[~use]
    y_dataRest = y_dataAll[~use]
    num = len(x_data)
    a = tf.Variable([2.])
    b = tf.Variable([0.])
    c = tf.Variable([0.])

    print(type(c))
    
    x = tf.placeholder(tf.float32, shape=(len(x_data)))
    y = tf.placeholder(tf.float32, shape=(len(y_data)))

    # 目的関数の設定。平均自乗誤差の最小化
    mserror = tf.reduce_mean(tf.square(model(a,b,c,x) - y) )
    # 最適化手法の設定。最急降下法　alpha はいわゆる学習係数　適当
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    #  つまり平均自乗誤差を最急降下で減少させる試みを train と名付ける
    train = optimizer.minimize(mserror)

    # 学習を始めるためにやっておかねばならないおまじない（変数の初期化）を init とおく, 
    init = tf.global_variables_initializer()
    # init = tf.initialize_all_variables()

    # セッションを生成し、init つまり変数の初期化を実行
    with tf.Session() as sess:
        sess.run(init)

        # モデル当てはめのステップを繰り返し実行
        for step in range(itr+1):
            sess.run(train, feed_dict = {x: x_data, y: y_data})
            if step % (itr/10) == 0: # 100回（繰り返し回数の10分の１）毎に推定された係数値を表示
                da = sess.run(a)[0]
                db = sess.run(b)[0]
                dc = sess.run(c)[0]
                er = sess.run(mserror,feed_dict = {x:x_data,y:y_data})
                print("%5d:(a,b,c,err) = (%10.4f,%10.4f, %10.4f, %10.4f)"%(step,da,db,dc,er) )

        A = sess.run(a)
        B = sess.run(b)
        C = sess.run(c)

    plt.figure(figsize=(12,10))
    plt.gca().set_aspect('equal',adjustable='box')
    plt.plot(x_data,y_data,".",color="green")
    plt.plot(x_dataRest,y_dataRest,".",color="gray")
    # plt.hold(True);
    xd = np.linspace(xmin, xmax, num)
    yd = 1.0/(1.0+np.exp(-A * (xd - C)*(1.0+np.exp(-B * (xd - C)))))
    plt.plot(xd,yd,"-",color='red')