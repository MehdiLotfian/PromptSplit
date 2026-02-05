"""
kernel_method.py
----------------
Implements kernel-based methods for similarity and feature analysis.
"""

import argparse
import numpy as np
import torch
import json
from pathlib import Path

class KernelMethod:
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
        parser = argparse.ArgumentParser(description='Kernel Method Eigenvalue Analysis')
        parser.add_argument('--test-root', type=str, required=True, help='root folder with test per-job subfolders')
        parser.add_argument('--ref-root', type=str, required=True, help='root folder with reference per-job subfolders')
        parser.add_argument('--prompt-dir', type=str, required=True, help='folder containing prompt .pt files (one per job)')
        parser.add_argument('--chunk-size', type=int, default=10, help='Number of images per prompt chunk')
        parser.add_argument('--sigma-img', type=float, default=1.0, help='Sigma for image features')
        parser.add_argument('--sigma-prompt', type=float, default=1.0, help='Sigma for prompt features')
        parser.add_argument('--eta', type=float, default=1.0, help='eta param used in KEN differential matrix')
        parser.add_argument('--output', type=str, default='kernel_results.json', help='Output JSON file for eigenvalues and KEN score')
        args = parser.parse_args()

        # Load embeddings
        X_images, _, _ = load_embeddings_from_dir(args.test_root)
        Y_images, _, _ = load_embeddings_from_dir(args.ref_root)
        prompt_files = {p.stem: torch.load(p, map_location='cpu') for p in Path(args.prompt_dir).glob('*.pt')}
        prompt_emb = list(prompt_files.values())[0]
        if isinstance(prompt_emb, dict) and 'embeddings' in prompt_emb:
            prompt_emb = prompt_emb['embeddings']
        if hasattr(prompt_emb, 'cpu'):
            prompt_emb = prompt_emb.cpu().numpy()
        if prompt_emb.ndim == 1:
            prompt_emb = prompt_emb.reshape(1, -1)
        X_prompts = np.tile(prompt_emb, (X_images.shape[0], 1))
        Y_prompts = np.tile(prompt_emb, (Y_images.shape[0], 1))

        # Kernel computation
        method = KernelMethod(sigma_img=args.sigma_img, sigma_prompt=args.sigma_prompt, eta=args.eta)
        n_x = X_images.shape[0]
        n_y = Y_images.shape[0]
        images_all = np.vstack([X_images, Y_images])
        prompts_all = np.vstack([X_prompts, Y_prompts])
        K = method.joint_rbf_kernel(images_all, prompts_all)
        K = 0.5 * (K + K.T)
        A = method.build_A_from_K(K, n_x, n_y)
        w, _ = np.linalg.eigh(A)
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
    Implements kernel-based methods for similarity, joint kernels, and eigenvalue analysis.
    """
    def __init__(self, sigma_img=1.0, sigma_prompt=1.0, eta=1.0):
        self.sigma_img = sigma_img
        self.sigma_prompt = sigma_prompt
        self.eta = eta

    def rbf_kernel(self, X, Y=None, sigma=None):
        """
        Computes the RBF (Gaussian) kernel matrix between X and Y.
        Args:
            X (np.ndarray): Data matrix of shape (n_samples_X, n_features)
            Y (np.ndarray, optional): Data matrix of shape (n_samples_Y, n_features)
            sigma (float, optional): Bandwidth parameter
        Returns:
            np.ndarray: Kernel matrix
        """
        if sigma is None:
            sigma = self.sigma_img
        if Y is None:
            Y = X
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-K / (2 * sigma ** 2))

    def joint_rbf_kernel(self, images, prompts):
        """
        Computes the product kernel: k((x,p),(x',p')) = k_img(x,x') * k_prompt(p,p')
        """
        K_img = self.rbf_kernel(images, sigma=self.sigma_img)
        K_pr = self.rbf_kernel(prompts, sigma=self.sigma_prompt)
        return K_img * K_pr

    def build_A_from_K(self, K, n_x, n_y):
        N = n_x + n_y
        A = np.zeros((N, N), dtype=K.dtype)
        A[:n_x, :n_x] = K[:n_x, :n_x]
        A[n_x:, n_x:] = -self.eta * K[n_x:, n_x:]
        A = 0.5 * (A + A.T)
        return A

    def top_eigen_analysis(self, X_images, Y_images, X_prompts, Y_prompts, top_k=10):
        """
        Computes the top-k eigenvalues and eigenvectors of the joint kernel matrix.
        Returns:
            eigvals, eigvecs
        """
        n_x = X_images.shape[0]
        n_y = Y_images.shape[0]
        images_all = np.vstack([X_images, Y_images])
        prompts_all = np.vstack([X_prompts, Y_prompts])
        K = self.joint_rbf_kernel(images_all, prompts_all)
        K = 0.5 * (K + K.T)
        A = self.build_A_from_K(K, n_x, n_y)
        w, U = np.linalg.eigh(A)
        w = np.real(w)
        order = np.argsort(w)[::-1]
        w = w[order]
        U = U[:, order]
        return w[:top_k], U[:, :top_k]
