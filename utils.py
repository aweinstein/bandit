import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def save_figs_as_pdf(figs, fn):
    if isinstance(figs, list):
        pdf = PdfPages(fn)
        for f in figs:
            pdf.savefig(f)
        pdf.close()
    else:
        figs.savefig(fn, format='pdf')
    print('File %s created' % fn)

def softmax(Qs, beta):
    """Compute softmax probabilities for all actions."""
    num = np.exp(Qs * beta)
    den = np.exp(Qs * beta).sum()
    return num / den
