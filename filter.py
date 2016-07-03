import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def plot_H(b, a):
    ws = np.linspace(2*np.pi*3.7, 2*np.pi*8.6, 200)
    w, h = signal.freqs(b, a, ws)
    plt.semilogx(w/(2*np.pi), 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.show()

def filter_response(b, a, w):
    s = 1j * np.atleast_1d(w)
    h = np.abs(np.polyval(b, s) / np.polyval(a, s))
    h = 20 * np.log10(h)
    return h

def eval_filter(fcl, fch, feval_l=4, feval_h=8, N=4):
    b, a = signal.butter(N, [2*np.pi*fcl, 2*np.pi*fch], 'bandpass', 'true')
    Hl, Hh = filter_response(b, a, [2*np.pi*feval_l, 2*np.pi*feval_h])
    return Hl, Hh


if __name__ == '__main__':
    # Use a binary search to find a bandpass filter with gain -1.5db at 4 and 8
    # Hertz
    fl = 4
    f_sup = fl
    f_inf = fl / 2
    for _ in range(200):
        fh = 32 / fl
        Hl, Hh = eval_filter(fl, fh)
        print('fl: {:.3f}, fh: {:.3f}, Hl: {:.2f}, Hh: {:.2f}'.format(fl, fh, Hl, Hh))
        if np.abs(Hl - (-1.5)) < 1e-2:
            break
        if Hl > -1.5:
            f_inf = fl
        else:
            f_sup = fl
        fl = (f_inf + f_sup) / 2

    N = 4
    b, a = signal.butter(N, [2*np.pi*fl, 2*np.pi*fh], 'bandpass', 'true')
    plt.close('all')
    plot_H(b, a)
