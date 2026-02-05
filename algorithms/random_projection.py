"""
random_projection.py
--------------------
Implements the Random Projection method for dimensionality reduction and kernel approximation.
"""


import argparse
import numpy as np
import torch
import json
from pathlib import Path


class RandomProjection:
        @staticmethod
        def compute_rff_features(data_img, data_prompt, omega_img, omega_prompt):
            proj_img = data_img @ omega_img
            proj_prompt = data_prompt @ omega_prompt
            proj_joint = proj_img + proj_prompt
            cos_feat = np.cos(proj_joint)
            sin_feat = np.sin(proj_joint)
            r = omega_img.shape[1]
            return np.hstack([cos_feat, sin_feat]) / np.sqrt(r)

    def load_embeddings_from_dir(root_dir, max_per_job=None):
        all_embeddings = []
        job_names = []
        job_counts = []
        root = Path(root_dir)
        for job in sorted([p.name for p in root.iterdir() if p.is_dir()]):
            emb_path_pt = root / job / "embeddings.pt"
            emb_path_npz = root / job / "embeddings.npz"
            if emb_path_pt.exists():
                data = torch.load(emb_path_pt, map_location='cpu')
            elif emb_path_npz.exists():
                _npz = __import__('numpy').load(emb_path_npz, allow_pickle=True)
                files = list(_npz.files)
                if 'embeddings' in files:
                    data = _npz['embeddings']
                elif len(files) == 1:
                    data = _npz[files[0]]
                else:
                    data = {k: _npz[k] for k in files}
            else:
                continue
            if isinstance(data, dict) and 'embeddings' in data:
                emb = data['embeddings']
            else:
                emb = data
            if isinstance(emb, dict):
                if 'embeddings' in emb:
                    emb = emb['embeddings']
                else:
                    raise RuntimeError(f"Unexpected dict structure in embeddings.pt for job={job}")
            if hasattr(emb, 'cpu'):
                emb = emb.cpu().numpy()
            elif isinstance(emb, np.ndarray):
                emb = emb
            else:
                emb = np.asarray(emb)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if max_per_job is not None and emb.shape[0] > int(max_per_job):
                emb = emb[:int(max_per_job)]
            all_embeddings.append(emb)
            job_names.extend([job] * emb.shape[0])
            job_counts.append({'job': job, 'count': emb.shape[0]})
        if not all_embeddings:
            raise RuntimeError(f"No embeddings found under {root_dir}")
        stacked = np.vstack(all_embeddings)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        stacked = stacked / norms
        return stacked, job_names, job_counts

    def main():
        parser = argparse.ArgumentParser(description='Random Projection RFF baseline')
        parser.add_argument('--test-root', type=str, required=True, help='root folder with test per-job subfolders')
        parser.add_argument('--ref-root', type=str, required=True, help='root folder with reference per-job subfolders')
        parser.add_argument('--prompt-dir', type=str, required=True, help='folder containing prompt .pt files (one per job)')
        parser.add_argument('--chunk-size', type=int, default=10, help='Number of images per prompt chunk')
        parser.add_argument('--r', type=int, default=2000, help='RFF dimension for RFF')
        parser.add_argument('--sigma-img', type=float, default=1.0, help='Sigma for image features')
        parser.add_argument('--sigma-prompt', type=float, default=1.0, help='Sigma for prompt features')
        parser.add_argument('--eta', type=float, default=1.0, help='eta param used in KEN differential matrix')
        parser.add_argument('--output', type=str, default='rff_results.json', help='Output JSON file for eigenvalues and KEN score')
        args = parser.parse_args()

        # Load embeddings
        X_images, _, _ = load_embeddings_from_dir(args.test_root)
        Y_images, _, _ = load_embeddings_from_dir(args.ref_root)
        # For simplicity, use same prompt embeddings for all (extend as needed)
        prompt_files = {p.stem: torch.load(p, map_location='cpu') for p in Path(args.prompt_dir).glob('*.pt')}
        # Use first prompt for all
        prompt_emb = list(prompt_files.values())[0]
        if isinstance(prompt_emb, dict) and 'embeddings' in prompt_emb:
            prompt_emb = prompt_emb['embeddings']
        if hasattr(prompt_emb, 'cpu'):
            prompt_emb = prompt_emb.cpu().numpy()
        if prompt_emb.ndim == 1:
            prompt_emb = prompt_emb.reshape(1, -1)
        X_prompts = np.tile(prompt_emb, (X_images.shape[0], 1))
        Y_prompts = np.tile(prompt_emb, (Y_images.shape[0], 1))

        # RFF computation
        rng = np.random.RandomState(42)
        d_img = X_images.shape[1]
        d_prompt = X_prompts.shape[1]
        omega_img_X = rng.normal(0, 1.0 / args.sigma_img, size=(d_img, args.r))
        omega_img_Y = rng.normal(0, 1.0 / args.sigma_img, size=(d_img, args.r))
        omega_prompt = rng.normal(0, 1.0 / args.sigma_prompt, size=(d_prompt, args.r))
        Z_X = RandomProjection.compute_rff_features(X_images, X_prompts, omega_img_X, omega_prompt)
        Z_Y = RandomProjection.compute_rff_features(Y_images, Y_prompts, omega_img_Y, omega_prompt)
        cov_X = Z_X.T @ Z_X / Z_X.shape[0]
        cov_Y = Z_Y.T @ Z_Y / Z_Y.shape[0]
        diff_cov = cov_X - args.eta * cov_Y
        diff_cov = 0.5 * (diff_cov + diff_cov.T)
        w, _ = np.linalg.eigh(diff_cov)
        w = np.real(w)
        positive = w[w > 1e-6]
        ken = float(0.0)
        if len(positive) > 0:
            sum_positive = positive.sum()
            p = positive / sum_positive
            entropy = -np.sum(p * np.log(p))
            ken = float(entropy * sum_positive)
        result = {
            'eigenvalues': w.tolist(),
            'ken_score': ken
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

    if __name__ == '__main__':
        main()
    """
    Implements Random Fourier Features (RFF) for kernel approximation and random projection for dimensionality reduction.
    """
    def __init__(self, input_dim, output_dim, sigma=1.0, random_seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.random_seed = random_seed
        self._rng = np.random.RandomState(self.random_seed)
        self.omega = self._rng.normal(0, 1.0 / self.sigma, size=(self.input_dim, self.output_dim))
        self.bias = self._rng.uniform(0, 2 * np.pi, size=(self.output_dim,))

    def transform(self, X):
        """
        Projects the input data X to a lower-dimensional space using random projection.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim)
        Returns:
            np.ndarray: Projected data of shape (n_samples, output_dim)
        """
        return np.dot(X, self.omega)

    def rff_features(self, X):
        """
        Computes Random Fourier Features for RBF kernel approximation.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim)
        Returns:
            np.ndarray: RFF features of shape (n_samples, 2 * output_dim)
        """
        proj = X @ self.omega + self.bias
        return np.hstack([np.cos(proj), np.sin(proj)]) / np.sqrt(self.output_dim)

    def fit_transform(self, X):
        return self.transform(X)
