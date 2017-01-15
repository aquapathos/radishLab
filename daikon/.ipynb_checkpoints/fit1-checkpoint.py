# TensorFlow による曲線あてはめ
import matplotlib
# %matplotlib nbagg
# matplotlib.use('nbagg') 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#  y = 1/（1+exp(-a (x-b)) という関係式を使って
# 実験用に仮想の x, y のペアデータを生成し、
# TensorFlow で回帰曲線を求めてみたい。

def fit(data,itr=1000,alpha=1.0):  # itr 繰り返しの回数、alpha 学習係数
    #
    (x_data,y_data) = data
    a = tf.Variable([0.])
    b = tf.Variable([0.])

    x = tf.placeholder(tf.float32, shape=(len(x_data)))
    y = tf.placeholder(tf.float32, shape=(len(y_data)))

    def model(a,b,x):  # 1/(1+exp(-a*(x-b))
        ret = tf.sub(x,b)
        ret = tf.exp(-tf.mul(a,ret))
        ret = tf.div(1.0,tf.add(1.0,ret))
        return ret

    # 目的関数の設定。平均自乗誤差の最小化
    mserror = tf.reduce_mean(tf.square(model(a,b,x) - y))
    # 最適化手法の設定。最急降下法　alpha はいわゆる学習係数　適当
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    #  つまり平均自乗誤差を最急降下で減少させる試みを train と名付ける
    train = optimizer.minimize(mserror)

    # 学習を始めるためにやっておかねばならないおまじない（変数の初期化）を init とおく, 
    init = tf.initialize_all_variables()

    # セッションを生成し、init つまり変数の初期化を実行
    with tf.Session() as sess:
        sess.run(init)

        # モデル当てはめのステップを繰り返し実行
        for step in range(itr+1):
            sess.run(train, feed_dict = {x: x_data, y: y_data})
            if step % (itr/10) == 0: # 100回（繰り返し回数の10分の１）毎に推定された係数値を表示
                da = sess.run(a)[0]
                db = sess.run(b)[0]
                er = sess.run(mserror,feed_dict = {x:x_data,y:y_data})
                print("%5d:(a,b,err) = (%10.4f,%10.4f, %10.4f)"%(step,da,db,er) )

        A = sess.run(a)
        B = sess.run(b)


    plt.gca().set_aspect('equal',adjustable='box')
    plt.plot(x_data,y_data,".")
    plt.hold(True);
    xd = np.linspace(0, 3, 100)
    yd = 1.0/(1.0+np.exp(-A * (xd - B)))
    plt.plot(xd,yd,"-")