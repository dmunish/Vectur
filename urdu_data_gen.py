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
    def __init__(self, font_path, font_size=100, sampling_rate=0.2, speed_scale=5.0):
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size
        self.sampling_rate = sampling_rate  # Time interval between points
        self.speed_scale = speed_scale      # Global multiplier for spacing

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
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        img = Image.new('L', (w + 100, h + 100), 0)
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), bidi_text, font=self.font, fill=255)
        
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
        skeleton = skeletonize(mask > 127).astype(np.uint8)
        nodes = np.argwhere(skeleton > 0)
        G = nx.Graph()
        for r, c in nodes:
            G.add_node((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    if (r + dr, c + dc) in G:
                        G.add_edge((r, c), (r + dr, c + dc))

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        all_vectors = []
        global_time = 0.0

        for comp in components:
            subgraph = G.subgraph(comp)
            start_node = min(comp, key=lambda n: (n[0], -n[1]))
            pixel_path = self._right_preference_dfs(subgraph, start_node)
            
            # Resample based on kinematic speed
            resampled_stroke, end_time = self._resample_path(pixel_path, global_time)
            all_vectors.append(resampled_stroke)
            global_time = end_time + 1.0 # Lift pause
            
        return all_vectors

    def _right_preference_dfs(self, G, start_node):
        path, visited = [], set()
        stack = [start_node]
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                path.append(u)
                neighbors = sorted(G.neighbors(u), key=lambda n: n[1], reverse=True)
                for v in neighbors:
                    if v not in visited: stack.append(v)
        return path

    def _resample_path(self, path, start_time):
        """
        Samples points at constant time intervals. 
        Distance between points = speed * dt.
        """
        path = np.array(path)
        if len(path) < 2: 
            return np.array([[path[0][1], path[0][0], start_time]]), start_time

        # Calculate cumulative distance along pixels
        diffs = np.diff(path, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        
        # Kinematic Speed Profile: Slow down at curves (approx)
        # For simplicity: higher local curvature = lower speed
        total_len = cum_dist[-1]
        
        # Create a temporal mapping: we want to sample at fixed 'self.sampling_rate'
        # Points further apart = higher velocity
        # New T array representing uniform clock ticks
        num_samples = int(total_len / self.speed_scale)
        if num_samples < 2: num_samples = 2
        
        # Interpolate X and Y coordinates based on the speed-weighted distance
        # To make speed controllable, we adjust the number of points taken over the length
        interp_dists = np.linspace(0, total_len, num_samples)
        
        new_x = np.interp(interp_dists, cum_dist, path[:, 1])
        new_y = np.interp(interp_dists, cum_dist, path[:, 0])
        times = start_time + np.arange(num_samples) * self.sampling_rate
        
        return np.column_stack((new_x, new_y, times)), times[-1]

    def save_visualizations(self, text, handwriting_img, vectors, prefix):
        # 1. Save Simulation
        cv2.imwrite(f"{prefix}_handwriting.png", handwriting_img)
        
        # 2. Vector Plot (Numbering instead of colors, No axes)
        fig, ax = plt.subplots(figsize=(12, 5))
        point_count = 0
        
        for stroke in vectors:
            ax.plot(stroke[:, 0], -stroke[:, 1], 'o', markersize=2, color='black', alpha=0.6)
            
            # Add numbering every 10 points
            for i in range(0, len(stroke), 10):
                ax.text(stroke[i, 0]+1, -stroke[i, 1]+1, str(point_count + i), 
                        fontsize=6, color='red', alpha=0.8)
            point_count += len(stroke)

        ax.set_aspect('equal')
        ax.axis('off') # Requirement 1: No axes
        plt.tight_layout()
        plt.savefig(f"{prefix}_vectors.png", bbox_inches='tight', pad_inches=0)
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
    word = "کتاب"
    mask = gen._text_to_mask(word)
    hw = gen.generate_handwriting_sim(mask)
    vecs = gen.get_motion_vectors(mask)
    gen.save_visualizations(word, hw, vecs, "output_sample")

if __name__ == "__main__":
    main()