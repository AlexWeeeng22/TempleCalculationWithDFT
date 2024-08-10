# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('=== Program Started ===')

    Ts = 1  # 采样时间
    fs = 1400  # 采样频率
    N = Ts * fs  # 采样点数
    # 在Ts内采样N个点
    xs = np.linspace(0, Ts, int(N))

    # 生成采样信号 由180Hz，390Hz和600Hz的正弦波叠加
    ys = 7.0 * np.sin(2 * np.pi * 180 * xs) + 2.8 * np.sin(2 * np.pi * 390 * xs) + 5.1 * np.sin(2 * np.pi * 600 * xs)

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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
