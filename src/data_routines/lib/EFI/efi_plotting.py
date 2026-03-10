import matplotlib.pyplot as plt

def plot_efi_eac_spec(efi,ax=None,cmap='jet',title=None,ylabel='Frequency (Hz)',
                      xlabel='Time (UTC)', zlim=(1e-12, 1e-10)):
    im=ax.pcolormesh(efi['eac']['ts2_l2_eac_packet_start'],
                     efi['eac']['Frequency'],
                     efi['eac']['ts2_l2_eac_x_spec'].T,
                     cmap=cmap, norm='log', vmin=zlim[0], vmax=zlim[1])
    cbax = ax.inset_axes([1.01,0,0.03,1],transform=ax.transAxes)
    cb = plt.colorbar(im, cax=cbax, label='$(V/m)^2/Hz$')

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


    return ax

def plot_efi_eac_ts(efi,ax=None,cmap='jet',title=None,ylabel='$E$ (V/m)',
                      xlabel='Time (UTC)'):
    pl=ax.plot(efi['eac']['Epoch'], efi['eac']['ts2_l2_eac'])

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


    return ax
