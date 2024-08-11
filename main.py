# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# git remote add origin https://github.com/AlexWeeeng22/TempleCalculationWithDFT.git
# git branch -M main
# git push -u origin main

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


def myDFT(ys, k, N, fs):
    '''
    :param ys:离散时域信号
    :param k:频域索引
    :param N:采样点数
    :param fs:采样信号
    :return:
    '''
    Xk = []
    for i in range(k):
        X = 0. + 0j
        for j in range(N):
            X += ys[j] * (np.cos(2 * np.pi / N * j * i) - 1j * np.sin(2 * np.pi / N * j * i))
        Xk.append(X)

    A = abs(np.array(Xk))  # 计算模值
    amp_x = A / N * 2  # 纵坐标变换
    label_x = np.linspace(0, int(N / 2) - 1, int(N / 2))  # 生成频率坐标
    amp = amp_x[0:int(N / 2)]  # 选取前半段计算结果即可，幅值  对称
    fs = fs  # 计算采样频率
    fre = label_x / N * fs  # 频率坐标变换

    return Xk, A, amp, fre

# 简单定义一个FFT函数
def myfft(x, t, fs):
    fft_x = fft(x)                                            # fft计算
    amp_x = abs(fft_x)/len(x)*2                               # 纵坐标变换  abs:求模长
    label_x = np.linspace(0,int(len(x)/2)-1,int(len(x)/2))    # 生成频率坐标
    amp = amp_x[0:int(len(x)/2)]                              # 选取前半段计算结果即可  对称
    # amp[0] = 0                                              # 可选择是否去除直流量信号
    fre = label_x/len(x)*fs                                   # 频率坐标变换
    pha = np.unwrap(np.angle(fft_x))                          # 计算相位角并去除2pi跃变
    return amp,fre,pha                                        # 返回幅度和频率



def rect_wave(x, freq, w):  # 生成频率为freq(unit/hour)，宽度为w(mins)（矩形波分布在频率中心两侧）的一个矩形波，x为自变量domain， 函数输出为应变量range
    T = 1/freq * 60 # mins/unit
    if (x % T <= (T + w/2)) & (x % T >= (T - w/2)):
        r = 1
    else:
        r = 0.0
    return r

fitLocationFreq = [0.04, 0.04, 0.07, 0.07, 0.07, 0.07,
                   1.56, 1.56, 1.56, 1.56, 0.78, 0.78, 0.3, 0.3, 0.45, 0.45,
                   0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.13, 0.13, 0.13, 0.13,
                   0.06, 0.06, 0.12, 0.12, 0.12, 0.12, 0.39, 0.39, 0.65, 0.65, 0.26, 0.26]

x0 = np.linspace(0, 960, 100) # [0, n] mins, resolution 3600
y0 = np.array([rect_wave(dx, fitLocationFreq[0], 10) for dx in x0])

for freq in fitLocationFreq:
    y1 = y0 + np.array([rect_wave(dx, freq, 10) for dx in x0])
    y0 = y1




plt.ylim(-0.1, 50)
plt.plot(x0, y1)
plt.show()

#进行离散傅里叶变换
Ts = 960  # 采样时间
fs = 1024  # 采样频率
N = Ts * fs  # 采样点数

y2 = y1
amp, fre, pha = myfft(y2, x0, fs)  # 调用scipy.fftpack里的fft
Xk, A, amp2, fre2 = myDFT(y2, int(N), int(N), fs)

plt.subplot(221)
plt.plot(x0, y2)
plt.title('OriSignal')
plt.xlabel('Time / s')
plt.ylabel('Intencity / cd')
plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('=== Program Started ===')

'''
Ts = 8  # 采样时间
fs = 100  # 采样频率
N = Ts * fs  # 采样点数

# 在Ts内采样N个点
xs = np.linspace(0, Ts, int(N))

# 生成采样信号 由180Hz正弦波叠加
ys = 7.0 * np.sin(2 * np.pi * 180 * xs)

amp, fre, pha = myfft(ys, xs, fs)  # 调用scipy.fftpack里的fft
Xk, A, amp2, fre2 = myDFT(ys, int(N), int(N), fs)

# 绘图
plt.subplot(221)
plt.plot(xs, ys)
plt.title('OriSignal')
plt.xlabel('Time / s')
plt.ylabel('Intencity / cd')

# 反傅里叶变换
ys390 = 2.8 * np.sin(2 * np.pi * 390 * xs)
H = np.zeros((int(N)))
H[390 - 50:390 + 50] = 1
H[1400 - 390 - 50:1400 - 390 + 50] = 1  # 将390Hz附近的频率获取
IFFT = ifft(H * Xk)
plt.subplot(223)
plt.plot(xs, IFFT, alpha=0.75, color='r')
plt.plot(xs, ys390, alpha=0.75, color='g')
plt.legend(['IFFT', 'ys390'])
plt.title('IFFT Filter')

plt.subplot(222)
plt.plot(fre, amp)
plt.title("'fft's Amplitute-Frequence-Curve")
plt.ylabel('Amplitute / a.u.')
plt.xlabel('Frequence / Hz')

plt.subplot(224)
plt.plot(fre2, amp2)
plt.title("myDFT's Amplitute-Frequence-Curve")
plt.ylabel('DFT Amplitute / a.u.')
plt.xlabel('DFT Frequence / Hz')
plt.show()


def rect_wave(x, c, c0):  # 起点为c0，宽度为c的矩形波
    if x >= (c + c0):
        r = 0.0
    elif x < c0:
        r = 0.0
    else:
        r = 1
    return r


x = np.linspace(-2, 4, 1000)
y = np.array([rect_wave(t, 2.0, -1.0) for t in x])
plt.ylim(-0.2, 1.2)
plt.plot(x, y)
plt.show()
'''






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
