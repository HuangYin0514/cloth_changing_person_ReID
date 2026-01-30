from matplotlib import pyplot as plt

# 自定义 IEEE 风格样式 --------------------------------
IEEE_STYLE = {
    "font.family": "sans-serif",
    "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(IEEE_STYLE)

# 定义常量 --------------------------------
DPI = 50
LATAX_DPI = 300
LINEWIDTH = 2
MARKERSIZE = 3
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "d"]
