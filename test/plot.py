import matplotlib.pyplot as plt
import numpy as np

# 启用LaTeX渲染

record_list = {
    'ISTA $\\ell_{\\frac{1}{2}}$': [
        -1.3332956435612975,
        -1.954847886967773,
        -2.363856895521723,
        -2.6774078107655725,
        -2.939360348119897,
        -3.17056169848996,
        -3.382350241393142,
        -3.5815185796677533,
        -3.7722434540218552,
        -3.957415042616045
    ],
    'FISTA $\\ell_{\\frac{1}{2}}$': [
        -1.339794013467428,
        -1.9682448131604935,
        -2.4899430293194613,
        -2.943451804244874,
        -3.3627563484313763,
        -3.773346762771016,
        -4.194302373541914,
        -4.640224411141361,
        -5.123179240853604,
        -5.654171721666686
    ],
    'LPA $\\ell_{\\frac{1}{2}}$': [
        0.8082846961617319,
        -2.8728041400892845,
        -3.432116615665288,
        -3.54480437283169,
        -5.1693069581526725,
        -8.605999599999063,
        -12.943094327323204,
        -13.975080358904195,
        -16.066789765689368,
        -20.395063034030414
    ]
}

# 设置足够长的样式列表
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5))]

plt.figure(figsize=(10, 6))

for index, (k, v) in enumerate(record_list.items()):
    plt.plot(
        np.arange(len(v)), 
        v, 
        color=colors[index % len(colors)],
        linestyle=styles[index % len(styles)],
        label=k,
        linewidth=2
    )

plt.legend(fontsize=12, loc='lower left')
plt.ylabel('NMSE (dB)', fontsize=14)
plt.xlabel('Iterations / Layers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('NMSE_comparison.pdf', dpi=300, bbox_inches='tight')
#plt.show()
