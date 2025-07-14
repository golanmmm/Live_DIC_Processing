import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

DOWNSAMPLE = 5
MAX_VECTOR = 0.005      # 5 מ״מ

# ---------- RealSense ----------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth , 1280, 720, rs.format.z16 , 30)
cfg.enable_stream(rs.stream.color , 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(cfg)

profile.get_device().first_depth_sensor().set_option(rs.option.visual_preset, 3)
align   = rs.align(rs.stream.color)
spatial = rs.spatial_filter()
temporal= rs.temporal_filter()
pc      = rs.pointcloud()

# ---------- ROI via OpenCV ----------
roi, drawing, ix, iy = None, False, -1, -1
def cb(event,x,y,flags,param):
    global roi,drawing,ix,iy
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing, ix, iy = True, x, y
        roi = (ix,iy,ix,iy)
    elif event==cv2.EVENT_MOUSEMOVE and drawing:
        roi = (ix,iy,x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (ix,iy,x,y)
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', cb)

print("בחר ROI עם העכבר ואז Enter להתחלה")

while True:
    f = pipeline.wait_for_frames()
    img = np.asarray(align.process(f).get_color_frame().get_data())
    if roi: cv2.rectangle(img,(roi[0],roi[1]),(roi[2],roi[3]),(0,255,255),2)
    cv2.imshow('Select ROI', img)
    k=cv2.waitKey(1)
    if k==13: break
    if k==27: exit()

x1,x2 = sorted([roi[0],roi[2]])
y1,y2 = sorted([roi[1],roi[3]])

# ---------- Open3D ----------
vis   = o3d.visualization.Visualizer()
vis.create_window('Live 3D Point Cloud',1280,720)
cloud = o3d.geometry.PointCloud(); vis.add_geometry(cloud)
prev  = None

try:
    while True:
        f  = pipeline.wait_for_frames()
        d  = temporal.process(spatial.process(align.process(f).get_depth_frame()))
        pts= pc.calculate(d); vtx=np.asarray(pts.get_vertices())
        h,w = 720,1280
        xyz = vtx.view(np.float32).reshape(h,w,3)
        xyz[~np.isfinite(xyz)] = np.nan

        sub = xyz[y1:y2, x1:x2].reshape(-1,3)
        sub = sub[np.isfinite(sub).all(1)]
        if sub.size==0: continue
        sub = sub[::DOWNSAMPLE]

        if prev is None:
            prev=sub.copy(); continue

        vec = sub - prev
        mag = np.linalg.norm(vec, axis=1)       # ← axis fixed
        mag = np.clip(mag,0,MAX_VECTOR)

        colors = np.zeros((len(mag),3))
        colors[:,0] = mag/MAX_VECTOR
        colors[:,2] = 1-mag/MAX_VECTOR

        cloud.points = o3d.utility.Vector3dVector(sub)
        cloud.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(cloud); vis.poll_events(); vis.update_renderer()
        prev = sub.copy()
finally:
    pipeline.stop(); vis.destroy_window()
