import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(nrows=1,ncols=2,
                      figsize=(9, 3.5)) # width, height
fig.suptitle('big title')

# left panel
ax[0].set_title('left panel title')
panel = ax[0].imshow(np.random.rand(30,30),origin='lower')
plt.colorbar(panel,ax=ax[0],label='axis 0 colorbar label')
#
ax[0].set_xlabel('distance (m)')
ax[0].set_xticks(np.arange(30)[::5], (np.arange(30)*150)[::5], rotation=90);
ax[0].set_ylabel('distance (m)')
ax[0].set_yticks(np.arange(30)[::5], (np.arange(30)*150)[::5]);

# right panel
ax[1].set_title('right panel title')
panel = ax[1].imshow(np.random.rand(90,90),origin='lower')
plt.colorbar(panel,ax=ax[1],label='axis 1 colorbar label')
#
ax[1].set_xlabel('distance (m)')
ax[1].set_xticks(np.arange(90)[::10], (np.arange(90)*50)[::10], rotation=90);
ax[1].set_ylabel('distance (m)')
ax[1].set_yticks(np.arange(90)[::10], (np.arange(90)*50)[::10]);

# 
plt.tight_layout()