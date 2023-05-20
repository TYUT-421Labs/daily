import gpxpy
import matplotlib.pyplot as plt
import pyproj


def plot_gpx_file(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    latitudes = []
    longitudes = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)

    # 将经纬度坐标转换为米
    proj = pyproj.Proj(proj='utm', zone=33, ellps='WGS84')
    x, y = proj(longitudes, latitudes)
    x = [item - min(x) for item in x]
    y = [item - min(y) for item in y]
    plt.plot(x, y, label=gpx_file)


# 读取并绘制第一个GPX文件
gpx_file1 = 'snapshot/track_drone/oval.gpx'
plot_gpx_file(gpx_file1)

# 读取并绘制第二个GPX文件
gpx_file2 = 'snapshot/track_people/oval.gpx'
plot_gpx_file(gpx_file2)

# 图例放在右上角
plt.legend(['people', 'drone'], loc='lower right', title='trace')
plt.ylabel('north(m)')
plt.xlabel('east(m)')

# 显示图形
plt.savefig("fig/trace.png", dpi=750, bbox_inches='tight')
plt.show()
