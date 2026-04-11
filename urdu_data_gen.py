import numpy as np
import cv2
import networkx as nx
import json
import imageio
import os
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

class UrduHandwritingGenerator:
    def __init__(self, font_path, font_size=100, sampling_rate=0.2, speed_scale=5.0, dpi=300):
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size
        self.sampling_rate = sampling_rate  # Time interval between points
        self.speed_scale = speed_scale      # Global multiplier for spacing
        self.dpi = dpi                      # Resolution for output plots

    def _distort_mask(self, mask):
        """Adds wavy edges and elastic distortions to the text mask."""
        rows, cols = mask.shape
        # Create a mesh grid for displacement
        map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # 1. Add low-frequency waves (Sine waves)
        map_x = map_x + 2.0 * np.sin(map_y / 10.0)
        map_y = map_y + 1.5 * np.cos(map_x / 15.0)
        
        # 2. Add high-frequency jitter (Hand tremble)
        noise = np.random.normal(0, 0.3, (rows, cols))
        map_x = (map_x + noise).astype(np.float32)
        map_y = (map_y + noise).astype(np.float32)

        distorted = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        return distorted.astype(np.uint8)

    def _text_to_mask(self, text):
        reshaped_text = reshape(text)
        bidi_text = get_display(reshaped_text)
        
        dummy_img = Image.new('L', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), bidi_text, font=self.font)
        # Calculate actual width and height
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Add generous padding to avoid cut-offs from text descenders, skew, or wavy distortions
        pad = 100
        img = Image.new('L', (w + 2*pad, h + 2*pad), 0)
        draw = ImageDraw.Draw(img)
        
        # Shift the drawing strictly by the bounding box offset to center it properly
        draw.text((pad - bbox[0], pad - bbox[1]), bidi_text, font=self.font, fill=255)
        
        mask = np.array(img)
        return self._distort_mask(mask)

    def generate_handwriting_sim(self, mask):
        """Dark ink against noisy paper."""
        rows, cols = mask.shape
        # Skewing
        pts1 = np.float32([[10, 10], [cols-10, 10], [10, rows-10]])
        pts2 = np.float32([[15, 20], [cols-5, 5], [5, rows-5]])
        M = cv2.getAffineTransform(pts1, pts2)
        distorted = cv2.warpAffine(mask, M, (cols, rows))

        bg = np.random.normal(242, 4, (rows, cols)).astype(np.uint8)
        text_indices = distorted > 127
        bg[text_indices] = np.random.randint(45, 75, size=np.sum(text_indices))
        return cv2.GaussianBlur(bg, (3, 3), 0)

    def get_motion_vectors(self, mask):
        # Bridge small gaps to connect disconnected curves that belong together
        binary_mask = (mask > 127).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        skeleton = skeletonize(closed_mask > 0).astype(np.uint8)
        nodes = np.argwhere(skeleton > 0)
        G = nx.Graph()
        for r, c in nodes:
            G.add_node((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    if (r + dr, c + dc) in G:
                        G.add_edge((r, c), (r + dr, c + dc))

        # Sort components from Right to Left explicitly
        components = sorted(nx.connected_components(G), key=lambda c: max(n[1] for n in c), reverse=True)
        all_vectors = []
        global_time = 0.0

        for comp in components:
            subgraph = G.subgraph(comp)
            # Find top-right-most node (balance top-to-bottom and right-to-left)
            start_node = min(comp, key=lambda n: n[0] - n[1])
            
            # Extract continuous paths to prevent unnatural straight-line jumps across the component
            strokes = self._extract_continuous_strokes(subgraph, start_node)
            
            for pixel_path in strokes:
                # Resample based on kinematic speed
                resampled_stroke, end_time = self._resample_path(pixel_path, global_time)
                if len(resampled_stroke) > 0:
                    all_vectors.append(resampled_stroke)
                global_time = end_time + 0.5 # Lift pause between strokes
            
            global_time += 1.0 # Extra lift pause between components
            
        return all_vectors

    def _extract_continuous_strokes(self, G, start_node):
        strokes = []
        visited = set()
        
        def trace_stroke(start):
            stroke = [start]
            visited.add(start)
            curr = start
            while True:
                unvisited_neighbors = [n for n in G.neighbors(curr) if n not in visited]
                if not unvisited_neighbors:
                    break
                # Right-preference: pick the unvisited neighbor furthest to the right
                next_node = sorted(unvisited_neighbors, key=lambda n: n[1], reverse=True)[0]
                stroke.append(next_node)
                visited.add(next_node)
                curr = next_node
            return stroke

        # Start the first stroke at the specified start node
        strokes.append(trace_stroke(start_node))
        
        # Continue starting new strokes for any branches we missed, preserving the graph topology
        unvisited_nodes = set(G.nodes) - visited
        while unvisited_nodes:
            # Prefer starting at a node adj to an already visited node to minimize leap distance
            next_start = None
            for n in unvisited_nodes:
                if any(nbr in visited for nbr in G.neighbors(n)):
                    next_start = n
                    break
            
            if next_start is None:
                # Disjoint sub-graph scenario (failsafe)
                next_start = min(unvisited_nodes, key=lambda n: n[0] - n[1])
                
            strokes.append(trace_stroke(next_start))
            unvisited_nodes = set(G.nodes) - visited
            
        return strokes

    def _resample_path(self, path, start_time):
        """
        Samples points at constant time intervals. 
        Implements the Two-Thirds Power Law for realistic kinematic speed:
        Velocity is proportional to the radius of curvature to the 1/3 power.
        """
        path = np.array(path, dtype=np.float32)
        
        if len(path) == 0:
            return np.empty((0, 3)), start_time
        if len(path) < 5: 
            # Too short to compute reliable curvature; assume constant speed
            times = start_time + np.arange(len(path)) * self.sampling_rate
            return np.column_stack((path[:, 1], path[:, 0], times)), times[-1]

        # 1. Smooth coordinate trajectory to compute stable derivatives
        window = min(15, len(path))
        kernel = np.ones(window) / window
        x_pad = np.pad(path[:, 1], (window//2, window//2), mode='edge')
        y_pad = np.pad(path[:, 0], (window//2, window//2), mode='edge')
        x_s = np.convolve(x_pad, kernel, mode='valid')
        y_s = np.convolve(y_pad, kernel, mode='valid')
        
        # Equalize length after convolution
        min_len = min(len(path), len(x_s))
        x_s = x_s[:min_len]
        y_s = y_s[:min_len]
        
        # 2. First and Second Derivatives
        dx = np.gradient(x_s)
        dy = np.gradient(y_s)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # 3. Compute Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        speed_sq = dx**2 + dy**2
        speed_sq[speed_sq < 1e-5] = 1e-5 # prevent division by zero
        k = np.abs(dx * ddy - dy * ddx) / (speed_sq**(1.5))
        
        # 4. Two-Thirds Power Law: Velocity v ~ (k + e)^(-1/3)
        # We add epsilon to k to clip maximum straight-line speed
        epsilon = 0.005
        v = (k + epsilon)**(-1/3)
        
        # Scale generic velocity contour up by user-defined speed_scale
        v = v * self.speed_scale
        
        # 5. Map spatial distance to time
        pts = np.column_stack((x_s, y_s))
        diffs = np.diff(pts, axis=0)
        ds = np.sqrt(np.sum(diffs**2, axis=1)) # length is min_len - 1
        
        # Average velocity for each segment
        v_segment = (v[:-1] + v[1:]) / 2.0
        
        dt = ds / v_segment
        dt = np.clip(dt, 1e-4, np.inf) # minimal viable time diff
        
        cum_time = np.concatenate(([0], np.cumsum(dt))) # length is min_len
        total_time = cum_time[-1]
        
        # 6. Interpolate points at constant sampling rate
        num_samples = int(total_time / self.sampling_rate)
        if num_samples < 2: 
            num_samples = 2
            
        target_times = np.linspace(0, total_time, num_samples)
        
        # Interpolate against original unsmoothed points to preserve sharp corners
        new_x = np.interp(target_times, cum_time, path[:min_len, 1])
        new_y = np.interp(target_times, cum_time, path[:min_len, 0])
        final_times = start_time + target_times
        
        return np.column_stack((new_x, new_y, final_times)), final_times[-1]

    def save_visualizations(self, text, handwriting_img, vectors, prefix):
        # 1. Save Simulation
        cv2.imwrite(f"{prefix}_handwriting.png", handwriting_img)
        
        # 2. Vector Plot (Numbering instead of colors, No axes)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=self.dpi)
        point_count = 0
        
        for stroke in vectors:
            ax.plot(stroke[:, 0], -stroke[:, 1], 'o', markersize=2, color='black', alpha=0.6)
            
            # Add numbering to each point
            for i in range(len(stroke)):
                ax.text(stroke[i, 0]+0.5, -stroke[i, 1]+0.5, str(point_count + i), 
                        fontsize=4, color='red', alpha=0.8)
            point_count += len(stroke)

        ax.set_aspect('equal')
        ax.axis('off') # Requirement 1: No axes
        plt.tight_layout()
        plt.savefig(f"{prefix}_vectors.png", bbox_inches='tight', pad_inches=0, dpi=self.dpi)
        plt.close()

        # 3. GIF Generation
        frames = []
        all_pts = np.vstack(vectors)
        for i in range(2, len(all_pts), max(1, len(all_pts)//40)):
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.scatter(all_pts[:i, 0], -all_pts[:i, 1], c='black', s=2)
            ax.set_xlim(np.min(all_pts[:,0])-20, np.max(all_pts[:,0])+20)
            ax.set_ylim(-np.max(all_pts[:,1])-20, -np.min(all_pts[:,1])+20)
            ax.axis('off')
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
            plt.close()
        imageio.mimsave(f"{prefix}_order.gif", frames, fps=10)

def main():
    # Usage
    gen = UrduHandwritingGenerator("NotoSansArabic-ExtraLight.ttf", speed_scale=8.0)
    word = "دانش"
    mask = gen._text_to_mask(word)
    hw = gen.generate_handwriting_sim(mask)
    vecs = gen.get_motion_vectors(mask)
    gen.save_visualizations(word, hw, vecs, "output_sample")

if __name__ == "__main__":
    main()