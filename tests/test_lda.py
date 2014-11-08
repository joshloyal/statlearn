def plot_lda(lda, X, y, y_pred, fig_index):

    splot = plt.subplot(2, 1, fig_index)
    tp = (y == y_pred) # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    # class 1: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')
    
    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')
    
    # means
    plt.plot(lda.means[0][0], lda.means[0][1], 'o', color='black', markersize=10)
    plt.plot(lda.means[1][0], lda.means[1][1], 'o', color='black', markersize=10)
    
    plot_lda_cov(lda, splot)
    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180* angle / np.pi      # convert to degrees

    # filled Gaussian at 2 standard deviation (95% confidence ellipse)
    s = 5.991 # P(s < 5.991) = 0.95 for a chi2 distribution
    ell = mpl.patches.Ellipse(mean, 2 * (s*v[0]) ** 0.5, 2 * (s*v[1]) ** 0.5,
                              180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())

def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means[0], lda.sigmaHat, 'red')
    plot_ellipse(splot, lda.means[1], lda.sigmaHat, 'blue')
