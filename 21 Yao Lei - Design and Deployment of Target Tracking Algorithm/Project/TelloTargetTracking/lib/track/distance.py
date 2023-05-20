import gpxpy
import matplotlib.pyplot as plt
import numpy as np
import datetime

def load_gpx_file(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    latitudes = []
    longitudes = []
    times = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)
                times.append(point.time)

    return latitudes, longitudes, times

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半径，单位为千米

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    return d

gpx_file1 = 'snapshot/track_drone/oval.gpx'
gpx_file2 = 'snapshot/track_people/oval.gpx'

lat1, lon1, time1 = load_gpx_file(gpx_file1)
lat2, lon2, time2 = load_gpx_file(gpx_file2)

time1 = np.array([(t - time1[0]).total_seconds() for t in time1])
time2 = np.array([(t - time2[0]).total_seconds() for t in time2])

time_max = int(min(time1[-1], time2[-1]))

lat1 = np.interp(np.arange(time_max), time1, lat1)
lon1 = np.interp(np.arange(time_max), time1, lon1)
lat2 = np.interp(np.arange(time_max), time2, lat2)
lon2 = np.interp(np.arange(time_max), time2, lon2)

distances = haversine(lat1, lon1, lat2, lon2)

plt.plot(np.arange(time_max), distances)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')

plt.savefig("fig/distance.png", dpi=750, bbox_inches='tight')
plt.show()
