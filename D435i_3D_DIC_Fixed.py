
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

def processing_worker(pipe, config):
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, config['resolution'][0], config['resolution'][1], rs.format.z16, config['fps'])
        cfg.enable_stream(rs.stream.color, config['resolution'][0], config['resolution'][1], rs.format.bgr8, config['fps'])
        pipeline.start(cfg)

        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        decimation = rs.decimation_filter()
        hole_filling = rs.hole_filling_filter()

        pc = rs.pointcloud()
        align = rs.align(rs.stream.color)
        prev_gray = None
        frame_counter = 0

        while True:
            if pipe.poll():
                msg = pipe.recv()
                if msg == 'STOP':
                    break
                else:
                    config.update(msg)

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if config['filters_enabled']:
                spatial.set_option(rs.option.filter_magnitude, config['spatial_strength'])
                temporal.set_option(rs.option.filter_smooth_alpha, config['temporal_alpha'])
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = decimation.process(depth_frame)
                depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            mask = np.isfinite(verts).all(axis=1)
            verts_masked = verts[mask]

            if verts_masked.shape[0] < 100:
                continue

            disp_stats = {'min': 0, 'max': 0, 'mean': 0}
            H, W = gray.shape
            if prev_gray is not None and verts.shape[0] == H * W:
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    dx = flow[..., 0].flatten()
                    dy = flow[..., 1].flatten()
                    dz = verts[:, 2]

                    disp_xy = np.sqrt(dx ** 2 + dy ** 2)
                    if config['mode'] == 'X-Displacement':
                        disp = np.abs(dx)
                    elif config['mode'] == 'Y-Displacement':
                        disp = np.abs(dy)
                    elif config['mode'] == 'Z-Displacement':
                        disp = np.abs(dz)
                    elif config['mode'] == 'Equivalent Strain':
                        disp = np.sqrt(dx ** 2 + dy ** 2)
                    elif config['mode'] == 'Total Displacement':
                        disp = np.sqrt(disp_xy ** 2 + dz ** 2)
                    else:
                        disp = disp_xy

                    disp = disp[mask]
                    disp_stats['min'] = float(np.min(disp))
                    disp_stats['max'] = float(np.max(disp))
                    disp_stats['mean'] = float(np.mean(disp))

                    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                    colors = cmap.reshape(-1, 3) / 255.0

                except Exception as e:
                    print(f'[Worker Warning] {str(e)}')
                    colors = np.full((verts_masked.shape[0], 3), [0.3, 0.3, 0.3])
            else:
                colors = np.full((verts_masked.shape[0], 3), [0.3, 0.3, 0.3])

            prev_gray = gray.copy()
            frame_counter += 1

            if frame_counter % config['mesh_update_rate'] == 0:
                reduced_verts = verts_masked[::config['downsample_factor']]
                reduced_colors = colors[::config['downsample_factor']]
                pipe.send(('DATA', reduced_verts, reduced_colors, disp_stats))

        pipeline.stop()
    except Exception as e:
        pipe.send(('ERROR', str(e)))

# The rest of the GUI class would be here... (truncated for space)
