import numpy as np
import cv2
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from skimage.morphology import skeletonize
import imageio
import matplotlib.pyplot as plt

class UrduHandwritingGenerator:
    def __init__(self, font_path, font_size=100):
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size

    def _text_to_mask(self, text):
        """Renders Urdu text to a high-res binary mask."""
        # 1. Handle Urdu RTL and Ligatures
        reshaped_text = reshape(text)
        bidi_text = get_display(reshaped_text)
        
        # 2. Create canvas
        dummy_img = Image.new('L', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        w, h = draw.textbbox((0, 0), bidi_text, font=self.font)[2:]
        
        img = Image.new('L', (w + 40, h + 40), 0)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), bidi_text, font=self.font, fill=255)
        return np.array(img)

    def generate_handwriting_sim(self, mask):
        """Creates a distorted, noisy PNG simulation of handwriting."""
        # Skewing / Affine Transform
        rows, cols = mask.shape
        pts1 = np.float32([[5, 5], [cols - 5, 5], [5, rows - 5]])
        pts2 = np.float32([[0, 10], [cols, 0], [10, rows]])
        M = cv2.getAffineTransform(pts1, pts2)
        distorted = cv2.warpAffine(mask, M, (cols, rows))

        # Add Noise and Background
        bg = np.random.normal(240, 5, (rows, cols)).astype(np.uint8)
        text_indices = distorted > 127
        bg[text_indices] = np.random.randint(40, 80, size=np.sum(text_indices))
        
        # Ink bleed effect
        bg = cv2.GaussianBlur(bg, (3, 3), 0)
        return bg

    def get_motion_vectors(self, mask):
        """Extracts the spine and generates time-encoded vectors."""
        # 1. Skeletonize to get 1-pixel spine
        skeleton = skeletonize(mask > 127).astype(np.uint8)
        
        # 2. Convert pixels to Graph
        nodes = np.argwhere(skeleton > 0)
        G = nx.Graph()
        for r, c in nodes:
            G.add_node((r, c))
            # Check 8-neighbors to build edges
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    if (r + dr, c + dc) in G:
                        G.add_edge((r, c), (r + dr, c + dc))

        # 3. Traversal Logic
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        all_vectors = [] # List of (x, y, t)
        current_time = 0.0

        for i, comp in enumerate(components):
            subgraph = G.subgraph(comp)
            # Start: Topmost-Rightmost (Min Y, Max X)
            start_node = min(comp, key=lambda n: (n[0], -n[1]))
            
            # DFS Traversal with "Right-first" preference
            path = self._right_preference_dfs(subgraph, start_node)
            
            # 4. Kinematics (Sigma-Lognormal / Power Law)
            vectors, end_time = self._apply_kinematics(path, current_time)
            all_vectors.append(vectors)
            current_time = end_time + 0.5 # Pen-lift delay
            
        return all_vectors

    def _right_preference_dfs(self, G, start_node):
        path = []
        visited = set()
        stack = [start_node]
        
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                path.append(u)
                # Sort neighbors: preferring right (higher column index)
                neighbors = sorted(G.neighbors(u), key=lambda n: n[1], reverse=True)
                for v in neighbors:
                    if v not in visited:
                        stack.append(v)
        return path

    def _apply_kinematics(self, path, start_time):
        """Calculates timestamps using a curvature-velocity power law."""
        vectors = []
        t = start_time
        for i in range(len(path)):
            if i == 0:
                vectors.append((path[i][1], path[i][0], t))
                continue
            
            # Dead simple curvature approximation: 
            # Speed is constant for this MVP, but hook for Power Law:
            # v = k * (curvature)**-1/3
            # Here we use a distance-based dt for simplicity
            dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            dt = dist * 0.05 # Adjust for "speed"
            t += dt
            vectors.append((path[i][1], path[i][0], t))
            
        return np.array(vectors), t

    def save_visualizations(self, text, handwriting_img, vectors, filename_prefix):
        # 1. Save Handwriting Simulation
        cv2.imwrite(f"{filename_prefix}_handwriting.png", handwriting_img)
        
        # 2. Save Vector Plot (Color = Time)
        plt.figure(figsize=(10, 4))
        for stroke in vectors:
            plt.scatter(stroke[:, 0], -stroke[:, 1], c=stroke[:, 2], cmap='viridis', s=2)
        plt.axis('equal')
        plt.title(f"Motion Vectors for: {text}")
        plt.savefig(f"{filename_prefix}_vectors.png")
        plt.close()

        # 3. Generate GIF
        frames = []
        full_path = np.vstack(vectors)
        for i in range(0, len(full_path), 5): # Step by 5 for speed
            fig, ax = plt.subplots()
            ax.scatter(full_path[:i, 0], -full_path[:i, 1], c='black', s=1)
            ax.set_xlim(np.min(full_path[:,0])-10, np.max(full_path[:,0])+10)
            ax.set_ylim(-np.max(full_path[:,1])-10, -np.min(full_path[:,1])+10)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close()
        imageio.mimsave(f"{filename_prefix}_order.gif", frames, fps=10)

# --- High Level Module ---
def process_urdu_file(file_path, font_path):
    gen = UrduHandwritingGenerator(font_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for idx, line in enumerate(lines):
        print(f"Processing line {idx+1}: {line}")
        mask = gen._text_to_mask(line)
        hw_img = gen.generate_handwriting_sim(mask)
        vectors = gen.get_motion_vectors(mask)
        gen.save_visualizations(line, hw_img, vectors, f"output_line_{idx}")

if __name__ == "__main__":
    # Ensure you have the font file in the directory
    process_urdu_file("urdu_text.txt", "NotoSansArabic-ExtraLight.ttf")