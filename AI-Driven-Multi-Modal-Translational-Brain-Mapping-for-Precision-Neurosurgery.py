"""
AI-Driven Multi-Modal Translational Brain Mapping for Precision Neurosurgery
Full end-to-end pipeline 
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img

# Try import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# --------------------------- Utilities ---------------------------------

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header


def save_nifti(data, affine, path):
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --------------------------- Preprocessing -----------------------------

def normalize_volume(vol, clip_percentile=(0.5, 99.5)):
    pmin, pmax = np.percentile(vol, clip_percentile)
    vol = np.clip(vol, pmin, pmax)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    return vol

# --------------------------- Dataset -----------------------------------
class MultiModalBrainDataset(Dataset):
    def __init__(self, cases, channels=['T1','T2','FLAIR'], patch_size=(128,128,128)):
        self.cases = cases
        self.channels = channels
        self.patch_size = patch_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []
        for ch in self.channels:
            p = os.path.join(case, f"{ch}.nii.gz")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            data, affine, header = load_nifti(p)
            data = normalize_volume(data)
            imgs.append(data)
        img = np.stack(imgs, axis=0).astype(np.float32)

        mask_path = os.path.join(case, 'mask.nii.gz')
        if os.path.exists(mask_path):
            mask, _, _ = load_nifti(mask_path)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros(img.shape[1:], dtype=np.uint8)

        img = center_crop_or_pad(img, self.patch_size)
        mask = center_crop_or_pad(mask[None], self.patch_size)[0]

        return torch.from_numpy(img), torch.from_numpy(mask).long(), case


def center_crop_or_pad(arr, target_shape):
    if arr.ndim == 4:
        C = arr.shape[0]
        out = np.zeros((C,)+tuple(target_shape), dtype=arr.dtype)
        for c in range(C):
            out[c] = center_crop_or_pad(arr[c], target_shape)
        return out
    assert arr.ndim == 3
    out = np.zeros(tuple(target_shape), dtype=arr.dtype)
    in_shape = arr.shape
    iz, iy, ix = in_shape
    tz, ty, tx = target_shape
    cz = max(0,(iz - tz)//2)
    cy = max(0,(iy - ty)//2)
    cx = max(0,(ix - tx)//2)
    cropped = arr[cz:cz+tz, cy:cy+ty, cx:cx+tx]
    oz, oy, ox = cropped.shape
    out[:oz, :oy, :ox] = cropped
    return out

# --------------------------- Model -------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2, diffZ//2, diffZ - diffZ//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, base_filters=32):
        super().__init__()
        self.inc = ConvBlock(in_ch, base_filters)
        self.down1 = Down(base_filters, base_filters*2)
        self.down2 = Down(base_filters*2, base_filters*4)
        self.down3 = Down(base_filters*4, base_filters*8)
        self.up1 = Up(base_filters*8, base_filters*4)
        self.up2 = Up(base_filters*4, base_filters*2)
        self.up3 = Up(base_filters*2, base_filters)
        self.outc = nn.Conv3d(base_filters, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

# --------------------------- Loss --------------------------------------
def dice_loss(pred, target, smooth=1e-5):
    probs = torch.softmax(pred, dim=1)[:,1]
    targetf = target.float()
    inter = (probs * targetf).sum()
    union = probs.sum() + targetf.sum()
    loss = 1 - (2. * inter + smooth) / (union + smooth)
    return loss

# --------------------------- Training ----------------------------------

def train(model, dataloader, device, epochs=10, lr=1e-4, save_path='model.pth'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks, case in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            ce = nn.CrossEntropyLoss()(preds, masks)
            dloss = dice_loss(preds, masks)
            loss = ce + dloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"{save_path}.epoch{epoch+1}")
    torch.save(model.state_dict(), save_path)
    return model

# --------------------------- Inference ---------------------------------

def predict_volume(model, vol_tensor, device):
    model.eval()
    with torch.no_grad():
        x = vol_tensor.to(device).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
        return probs

# --------------------------- fMRI Connectivity -------------------------

def extract_roi_time_series(fmri_img_path, atlas_img_path, confounds=None):
    masker = NiftiLabelsMasker(labels_img=atlas_img_path, standardize=True)
    time_series = masker.fit_transform(fmri_img_path, confounds=confounds)
    return time_series, masker


def compute_connectivity(time_series, method='correlation'):
    if method == 'correlation':
        return np.corrcoef(time_series.T)
    else:
        raise NotImplementedError

# --------------------------- Fusion ------------------------------------

def tumor_to_rois(tumor_prob_map, masker, threshold=0.5):
    atlas_img = masker.labels_img_
    atlas_data = atlas_img.get_fdata().astype(int)
    rois = np.unique(atlas_data)
    rois = rois[rois!=0]
    roi_scores = {}
    for roi in rois:
        mask = (atlas_data == roi)
        if mask.sum() == 0:
            roi_scores[int(roi)] = 0.0
            continue
        overlap = tumor_prob_map[mask].mean()
        roi_scores[int(roi)] = float(overlap)
    sorted_rois = sorted(roi_scores.items(), key=lambda x: -x[1])
    return sorted_rois

# --------------------------- Explainability ----------------------------

def compute_gradcam_3d(model, input_tensor, target_class=1, device='cpu'):
    model.to(device)
    model.eval()
    # register hooks
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    def save_gradient(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0].detach()
        return hook

    # find last conv layer
    last_conv = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv3d):
            last_conv = m
            break
    if last_conv is None:
        raise RuntimeError('No Conv3d found for Grad-CAM')

    h1 = last_conv.register_forward_hook(save_activation('last'))
    h2 = last_conv.register_full_backward_hook(save_gradient('last'))

    x = input_tensor.to(device).unsqueeze(0).requires_grad_(True)
    logits = model(x)
    score = logits[:, target_class].mean()
    model.zero_grad()
    score.backward()

    act = activations['last'][0]  # C, Z, Y, X
    grad = gradients['last'][0]
    weights = grad.mean(axis=(1,2,3), keepdims=True)
    cam = (weights * act).sum(axis=0)
    cam = np.maximum(cam.cpu().numpy(), 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove(); h2.remove()
    return cam

# ----------------------- Integrated Gradients --------------------------

def integrated_gradients(model, input_tensor, target_class=1, baseline=None, steps=50, device='cpu'):
    """Compute Integrated Gradients for a single input_tensor (C,Z,Y,X)
    baseline: same shape as input, if None uses zeros
    returns IG attribution map with same spatial dims (Z,Y,X)
    """
    model.to(device)
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    assert input_tensor.shape == baseline.shape

    scaled_inputs = [(baseline + float(i)/steps * (input_tensor - baseline)).unsqueeze(0).to(device) for i in range(1, steps+1)]
    total_grad = None
    for x in scaled_inputs:
        x.requires_grad_(True)
        logits = model(x)
        score = logits[:, target_class].sum()
        model.zero_grad()
        score.backward(retain_graph=False)
        grad = x.grad.detach().cpu().numpy()[0]  # C,Z,Y,X
        if total_grad is None:
            total_grad = grad
        else:
            total_grad += grad
    avg_grad = total_grad / steps
    # integrated grads: (input - baseline) * avg_grad summed over channels
    ig = (input_tensor.cpu().numpy() - baseline.cpu().numpy()) * avg_grad
    # sum over channels
    ig_map = ig.sum(axis=0)
    # take absolute and normalize for visualization
    ig_map = np.abs(ig_map)
    ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    return ig_map

# --------------------------- SHAP (voxel-level) ------------------------

def compute_shap_voxelwise(model, background_tensors, input_tensor, batch_size=8, device='cpu'):
    """Compute approximate voxel-wise SHAP using shap.GradientExplainer.
    background_tensors: numpy array (B, C, Z, Y, X)
    input_tensor: numpy array (1, C, Z, Y, X)
    returns shap_vals of same shape as input (C, Z, Y, X)
    Note: SHAP GradientExplainer returns sums across channels; we adapt to voxel-level by
    requesting per-voxel contributions. This is heavy – use small background.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError('shap not installed')
    device = device
    def f(x):
        x_t = torch.from_numpy(x).float().to(device)
        with torch.no_grad():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)[:,1]
        return probs.cpu().numpy()

    # Create explainer (GradientExplainer prefers torch models or functions)
    explainer = shap.GradientExplainer(f, background_tensors)
    # shap expects input with batch dim
    shap_vals = explainer.shap_values(input_tensor, nsamples='auto')
    # shap_vals shape: (1, C, Z, Y, X) or list depending on version
    # Convert to numpy array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return shap_vals[0]  # remove batch dim

# --------------------- XAI Visualization Utilities ---------------------

def make_publication_figure(t1_volume, tumor_prob, xai_map, slice_indices=None, save_path=None, dpi=300, cmap_overlay='hot'):
    """Create a 3-panel figure (axial, coronal, sagittal) overlaying tumor_prob and xai_map on top of T1.
    - t1_volume: 3D array (Z,Y,X)
    - tumor_prob: 3D array same shape
    - xai_map: 3D array same shape (normalized 0-1)
    - slice_indices: dict with keys 'axial','coronal','sagittal' giving slice index; if None use center
    - save_path: if provided, saves PNG and PDF
    """
    Z,Y,X = t1_volume.shape
    if slice_indices is None:
        slice_indices = {'axial': Z//2, 'coronal': Y//2, 'sagittal': X//2}

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    views = ['axial','coronal','sagittal']
    for ax, view in zip(axes, views):
        if view == 'axial':
            im = t1_volume[slice_indices['axial'],:,:]
            tp = tumor_prob[slice_indices['axial'],:,:]
            xm = xai_map[slice_indices['axial'],:,:]
        elif view == 'coronal':
            im = t1_volume[:,slice_indices['coronal'],:]
            tp = tumor_prob[:,slice_indices['coronal'],:]
            xm = xai_map[:,slice_indices['coronal'],:]
        else:  # sagittal
            im = t1_volume[:,:,slice_indices['sagittal']]
            tp = tumor_prob[:,:,slice_indices['sagittal']]
            xm = xai_map[:,:,slice_indices['sagittal']]

        ax.imshow(np.rot90(im), cmap='gray', interpolation='nearest')
        # overlay tumor prob as contour
        cont = ax.contour(np.rot90(tp), levels=[0.25,0.5,0.75], linewidths=1.0)
        # overlay XAI heatmap semi-transparent
        hm = ax.imshow(np.rot90(xm), cmap=cmap_overlay, alpha=0.5, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(view)
    fig.suptitle('Tumor probability contours (lines) & XAI saliency (overlay)')
    fig.tight_layout(rect=[0,0,1,0.95])

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        png = save_path + '.png'
        pdf = save_path + '.pdf'
        fig.savefig(png, dpi=dpi, bbox_inches='tight')
        fig.savefig(pdf, dpi=300, bbox_inches='tight')
        print(f'Saved figures: {png}, {pdf}')
    plt.close(fig)

# --------------------------- Visualization (3D) ------------------------

def voxels_from_probmap(probmap, affine=None, thresh=0.5):
    mask = (probmap >= thresh)
    coords = np.array(np.nonzero(mask)).T
    if affine is not None:
        vox = np.zeros((coords.shape[0], 4))
        vox[:,0:3] = coords[:,::-1]
        vox[:,3] = 1
        world = (affine @ vox.T).T[:,0:3]
        return coords, world
    return coords, coords

def plot_3d_tumor_and_rois(tumor_probmap, masker, atlas_img_path, affine, connectivity=None, top_k_rois=10, tumor_thresh=0.5):
    coords_vox, coords_world = voxels_from_probmap(tumor_probmap, affine, thresh=tumor_thresh)
    roi_scores = tumor_to_rois(tumor_probmap, masker, threshold=tumor_thresh)
    top_rois = [r for r,s in roi_scores[:top_k_rois] if s>0]
    atlas_img = nib.load(atlas_img_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_centroids = {}
    for roi in top_rois:
        mask = (atlas_data == roi)
        if mask.sum() == 0: continue
        idxs = np.array(np.nonzero(mask)).T
        vox = np.zeros((idxs.shape[0], 4))
        vox[:,0:3] = idxs[:,::-1]
        vox[:,3] = 1
        world = (affine @ vox.T).T[:,0:3]
        centroid = world.mean(axis=0)
        roi_centroids[roi] = centroid

    fig = go.Figure()
    if coords_world.shape[0] > 0:
        fig.add_trace(go.Scatter3d(x=coords_world[:,0], y=coords_world[:,1], z=coords_world[:,2], mode='markers', marker=dict(size=2), name='Tumor Voxels'))
    for roi,cent in roi_centroids.items():
        fig.add_trace(go.Scatter3d(x=[cent[0]], y=[cent[1]], z=[cent[2]], mode='markers+text', marker=dict(size=5), name=f'ROI {roi}', text=[str(roi)], textposition='top center'))

    if connectivity is not None and len(top_rois)>0:
        labels = masker.labels_
        for roi in top_rois:
            i = list(labels).index(roi) if roi in labels else None
            if i is None: continue
            row = connectivity[i]
            top_conn_idxs = np.argsort(-np.abs(row))[:5]
            for j in top_conn_idxs:
                if labels[j]==0 or labels[j] not in roi_centroids: continue
                a = roi_centroids[roi]
                b = roi_centroids[labels[j]]
                fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode='lines', line=dict(width=2), showlegend=False))

    fig.update_layout(scene=dict(aspectmode='data'), title='Tumor vs ROI 3D map')
    fig.show()

# --------------------------- CLI & Orchestration -----------------------

def parse_args():
    p = argparse.ArgumentParser(description='Translational Brain Mapping project (XAI expanded)')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--atlas', type=str, required=True)
    p.add_argument('--case_list', type=str, default=None)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run_train', action='store_true')
    p.add_argument('--run_infer', action='store_true')
    p.add_argument('--run_xai', action='store_true')
    p.add_argument('--save_figs', action='store_true')
    p.add_argument('--case_idx', type=int, default=0)
    p.add_argument('--save_dir', type=str, default='results')
    p.add_argument('--save_model', type=str, default='unet3d.pth')
    return p.parse_args()


def main():
    args = parse_args()
    if args.case_list and os.path.exists(args.case_list):
        with open(args.case_list) as f:
            cases = [l.strip() for l in f if l.strip()]
    else:
        cases = []
        for name in os.listdir(args.data_dir):
            p = os.path.join(args.data_dir, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p,'T1.nii.gz')):
                cases.append(p)
    print(f'Found {len(cases)} cases')
    if len(cases)==0:
        return

    dataset = MultiModalBrainDataset(cases, channels=['T1','T2','FLAIR'], patch_size=(128,128,128))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = UNet3D(in_ch=3, out_ch=2, base_filters=16)
    if args.run_train:
        model = train(model, dataloader, device=args.device, epochs=args.epochs, lr=args.lr, save_path=args.save_model)
    else:
        if os.path.exists(args.save_model):
            model.load_state_dict(torch.load(args.save_model, map_location=args.device))
            print('Loaded model', args.save_model)

    if args.run_infer:
        imgs, mask, case = dataset[args.case_idx]
        probs = predict_volume(model, imgs, device=args.device)
        t1_path = os.path.join(case, 'T1.nii.gz')
        t1_img = nib.load(t1_path)
        save_nifti(probs, t1_img.affine, os.path.join(case, 'tumor_prob.nii.gz'))

        fmri_path = os.path.join(case, 'fmri.nii.gz')
        if os.path.exists(fmri_path):
            time_series, masker = extract_roi_time_series(fmri_path, args.atlas)
            conn = compute_connectivity(time_series)
        else:
            masker = NiftiLabelsMasker(labels_img=args.atlas, standardize=True)
            conn = None

        if args.run_xai:
            run_xai_for_case(model, imgs, probs, t1_img.affine, args.atlas, masker, conn, args.save_dir, case_name=os.path.basename(case), device=args.device, save_figs=args.save_figs)
        else:
            plot_3d_tumor_and_rois(probs, masker, args.atlas, t1_img.affine, connectivity=conn)

# --------------------------- XAI Runner --------------------------------

def run_xai_for_case(model, input_tensor, tumor_prob, affine, atlas_path, masker, connectivity, save_dir='results', case_name='case', device='cpu', save_figs=True):
    ensure_dir(save_dir)
    model.to(device)
    model.eval()

    # Compute Grad-CAM
    print('Computing Grad-CAM...')
    gradcam_map = compute_gradcam_3d(model, input_tensor, target_class=1, device=device)

    # Compute Integrated Gradients
    print('Computing Integrated Gradients (this can be slow)...')
    baseline = torch.zeros_like(input_tensor)
    ig_map = integrated_gradients(model, input_tensor, target_class=1, baseline=baseline, steps=50, device=device)

    # Compute SHAP (if available) using small background (use other cases or small noise)
    shap_map = None
    if SHAP_AVAILABLE:
        print('Computing SHAP (may be memory heavy)...')
        # create background by taking a few slightly perturbed versions of the input
        bg = np.stack([input_tensor.numpy() + np.random.normal(0,0.01,size=input_tensor.shape) for _ in range(4)], axis=0)
        inp = input_tensor.numpy()[None]
        try:
            shap_vals = compute_shap_voxelwise(model, bg, inp, device=device)
            # shap_vals shape: C,Z,Y,X
            shap_map = np.abs(shap_vals).sum(axis=0)
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
        except Exception as e:
            print('SHAP failed:', e)
            shap_map = None

    # convert input_tensor (C,Z,Y,X) to T1 for background image for plotting
    t1_vol = input_tensor.numpy()[0]

    # produce publication-ready figures overlaying tumor_prob and each XAI map
    base_name = os.path.join(save_dir, case_name + '_xai')
    make_publication_figure(t1_vol, tumor_prob, gradcam_map, save_path=base_name + '_gradcam' if save_figs else None)
    make_publication_figure(t1_vol, tumor_prob, ig_map, save_path=base_name + '_ig' if save_figs else None)
    if shap_map is not None:
        make_publication_figure(t1_vol, tumor_prob, shap_map, save_path=base_name + '_shap' if save_figs else None)

    # also save combined 3D interactive if desired
    try:
        plot_3d_tumor_and_rois(tumor_prob, masker, atlas_path, affine, connectivity)
    except Exception as e:
        print('3D plot failed:', e)

# --------------------------- Entry point -------------------------------
if __name__ == '__main__':
    main()
