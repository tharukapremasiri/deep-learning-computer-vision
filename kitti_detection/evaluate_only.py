"""
=======================================================================
CET3013 Task 2 — EVALUATION ONLY
Loads saved model weights and generates all output images.
Run this AFTER training is complete (models A, B, C already saved).

All 3 models trained:
  kitti_model_A.pth  — Full Freeze (3 epochs)
  kitti_model_B.pth  — Partial Freeze (2 epochs from A)
  kitti_model_C.pth  — Temporal T=3 (1 epoch from B)

This script:
  1. Loads all 3 saved models
  2. Runs evaluation (IoU + mAP) on test set
  3. Generates kitti_map_comparison.png
  4. Generates kitti_predictions.png
=======================================================================
"""

# ===================================================================
# UPDATE THIS PATH
# ===================================================================
VIDEOS_ROOT = r'C:\Users\tharuka\OneDrive - McLarens Holdings Limited\Downloads\videos'

# ===================================================================
# SAME SETTINGS AS TRAINING SCRIPT
# ===================================================================
TRAIN_SEQS = ['VideoFive', 'VideoSeven', 'Video11', 'Video12', 'Video17']
VAL_SEQS   = ['Video15']
TEST_SEQS  = ['Video16']

MAX_TEST   = 50
IMG_W      = 800
IMG_H      = 256
BATCH_SIZE = 1
SCORE_THR  = 0.3
IOU_THR    = 0.5
NUM_CLASSES= 4

# ===================================================================
# IMPORTS
# ===================================================================
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

import os, glob, random, sys
import xml.etree.ElementTree as ET
from collections import defaultdict

CLASS_MAP    = {'Car':1,'Van':1,'Pedestrian':2,'Person':2,'Cyclist':3,
                'Tram':0,'Misc':0,'Truck':0}
KEEP_CLASSES = {'Car','Van','Pedestrian','Person','Cyclist'}
PRED_COLORS  = {1:'red', 2:'deepskyblue', 3:'lime'}
ID2NAME      = {1:'Car', 2:'Pedestrian', 3:'Cyclist'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*60)
print('CET3013 Task 2 — Evaluation Only (loading saved models)')
print('='*60)
print(f'Device: {device}')

# ===================================================================
# PARSERS (same as training script)
# ===================================================================
def parse_tracklet_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame_annotations = defaultdict(list)
    tracklets_elem = root.find('tracklets')
    if tracklets_elem is None: tracklets_elem = root
    for item in tracklets_elem.findall('item'):
        obj_type_elem = item.find('objectType')
        if obj_type_elem is None: continue
        obj_type = obj_type_elem.text.strip()
        if obj_type not in KEEP_CLASSES: continue
        h = float(item.find('h').text) if item.find('h') is not None else 1.5
        w = float(item.find('w').text) if item.find('w') is not None else 1.8
        l = float(item.find('l').text) if item.find('l') is not None else 4.0
        first_frame_elem = item.find('first_frame')
        first_frame = int(first_frame_elem.text) if first_frame_elem is not None else 0
        poses_elem = item.find('poses')
        if poses_elem is None: continue
        for frame_offset, pose in enumerate(poses_elem.findall('item')):
            frame_idx = first_frame + frame_offset
            tx_e,ty_e,tz_e = pose.find('tx'),pose.find('ty'),pose.find('tz')
            if tx_e is None or ty_e is None or tz_e is None: continue
            frame_annotations[frame_idx].append({
                'type':obj_type,
                'tx':float(tx_e.text),'ty':float(ty_e.text),'tz':float(tz_e.text),
                'h':h,'w':w,'l':l
            })
    return frame_annotations

def load_calib(calib_path):
    if not calib_path or not os.path.exists(calib_path): return None
    with open(calib_path) as f:
        for line in f:
            if line.startswith('P_rect_02:') or line.startswith('P2:'):
                vals = list(map(float, line.strip().split()[1:]))
                return np.array(vals).reshape(3,4)
    return None

def project_3d_to_2d(tx,ty,tz,h,w,l,P2,img_w=1242,img_h=375):
    corners = np.array([
        [l/2,w/2,0],[l/2,-w/2,0],[-l/2,w/2,0],[-l/2,-w/2,0],
        [l/2,w/2,h],[l/2,-w/2,h],[-l/2,w/2,h],[-l/2,-w/2,h],
    ])
    corners[:,0]+=tx; corners[:,1]+=ty; corners[:,2]+=tz
    cam = np.zeros_like(corners)
    cam[:,0]=-corners[:,1]; cam[:,1]=-corners[:,2]; cam[:,2]=corners[:,0]
    if np.all(cam[:,2]<=0): return None
    pts_h = np.hstack([cam,np.ones((8,1))])
    proj  = (P2 @ pts_h.T).T
    valid = proj[:,2]>0
    if not np.any(valid): return None
    proj=proj[valid]
    xs=proj[:,0]/proj[:,2]; ys=proj[:,1]/proj[:,2]
    x1=float(max(0,np.min(xs))); y1=float(max(0,np.min(ys)))
    x2=float(min(img_w,np.max(xs))); y2=float(min(img_h,np.max(ys)))
    if x2-x1<2 or y2-y1<2: return None
    return (x1,y1,x2,y2)

# ===================================================================
# DATASET (same as training script)
# ===================================================================
def find_image_folder(vpath):
    for root,dirs,files in os.walk(vpath):
        if os.path.basename(root)=='data' and 'image_02' in root:
            pngs=sorted(glob.glob(os.path.join(root,'*.png')))
            if pngs: return root,pngs
    return None,[]

def find_xml_label(vpath):
    for root,dirs,files in os.walk(vpath):
        for f in files:
            if 'tracklet_labels' in f: return os.path.join(root,f)
    return None

def find_calib_file(vpath):
    for root,dirs,files in os.walk(vpath):
        for f in files:
            if 'calib_cam_to_cam' in f: return os.path.join(root,f)
    return None

def discover_sequences(videos_root):
    sequences = {}
    video_folders = sorted([d for d in os.listdir(videos_root)
                             if os.path.isdir(os.path.join(videos_root,d))])
    for vname in video_folders:
        vpath = os.path.join(videos_root,vname)
        img_folder,img_list = find_image_folder(vpath)
        xml_path   = find_xml_label(vpath)
        calib_path = find_calib_file(vpath)
        if not img_list or not xml_path: continue
        frame_annots = parse_tracklet_xml(xml_path)
        sequences[vname]={
            'img_folder':img_folder,'img_list':img_list,
            'xml_path':xml_path,'calib_path':calib_path,
            'frame_annotations':frame_annots,
            'n_frames':len(img_list)
        }
    return sequences

class KITTIDataset(Dataset):
    def __init__(self, all_sequences, seq_names, T=1, max_samples=None):
        self.T = T
        self.samples = []
        for sname in seq_names:
            if sname not in all_sequences: continue
            seq    = all_sequences[sname]
            imgs   = seq['img_list']
            annots = seq['frame_annotations']
            P2     = load_calib(seq['calib_path'])
            for i,img_path in enumerate(imgs):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                try:    frame_idx = int(stem)
                except: frame_idx = i
                self.samples.append({
                    'img_path':img_path,'frame_idx':frame_idx,
                    'seq_i':i,'seq_len':len(imgs),'seq_imgs':imgs,
                    'annots':annots,'P2':P2
                })
        if max_samples and len(self.samples) > max_samples:
            step = len(self.samples) // max_samples
            self.samples = self.samples[::step][:max_samples]
        print(f'  Dataset: {len(self.samples)} samples | T={T}')

    def load_frame(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_W,IMG_H), Image.BILINEAR)
        return T.ToTensor()(img)

    def get_boxes(self, sample):
        frame_idx = sample['frame_idx']
        P2 = sample['P2']
        orig = Image.open(sample['img_path'])
        orig_w,orig_h = orig.size
        sx,sy = IMG_W/orig_w, IMG_H/orig_h
        objects = sample['annots'].get(frame_idx,[])
        boxes,labels = [],[]
        for obj in objects:
            cls_id = CLASS_MAP.get(obj['type'],0)
            if cls_id==0: continue
            if P2 is not None:
                result = project_3d_to_2d(
                    obj['tx'],obj['ty'],obj['tz'],
                    obj['h'],obj['w'],obj['l'],
                    P2,img_w=orig_w,img_h=orig_h)
            else: result = None
            if result is None: continue
            x1,y1,x2,y2 = result
            boxes.append([x1*sx,y1*sy,x2*sx,y2*sy])
            labels.append(cls_id)
        if boxes:
            return (torch.tensor(boxes,dtype=torch.float32),
                    torch.tensor(labels,dtype=torch.int64))
        return (torch.zeros((0,4),dtype=torch.float32),
                torch.zeros((0,),dtype=torch.int64))

    def __len__(self): return len(self.samples)

    def __getitem__(self,idx):
        sample  = self.samples[idx]
        seq_i   = sample['seq_i']
        seq_len = sample['seq_len']
        seq_imgs= sample['seq_imgs']
        if self.T==1:
            img_tensor = self.load_frame(sample['img_path'])
        else:
            half=self.T//2
            frames=[self.load_frame(seq_imgs[max(0,min(seq_len-1,seq_i+o))])
                    for o in range(-half,half+1)]
            img_tensor = torch.stack(frames).mean(0)
        boxes,labels = self.get_boxes(sample)
        return img_tensor, {'boxes':boxes,'labels':labels,'image_id':torch.tensor([idx])}

def collate_fn(batch): return tuple(zip(*batch))

# ===================================================================
# BUILD DATASETS
# ===================================================================
print('\nLoading dataset sequences...')
ALL_SEQUENCES = discover_sequences(VIDEOS_ROOT)

print('Building test datasets...')
test_dataset  = KITTIDataset(ALL_SEQUENCES, TEST_SEQS, T=1, max_samples=MAX_TEST)
test_temp     = KITTIDataset(ALL_SEQUENCES, TEST_SEQS, T=3, max_samples=MAX_TEST)

test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
test_loader_C = DataLoader(test_temp,    batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)

# ===================================================================
# MODEL BUILDER
# ===================================================================
def build_model(num_classes=4, freeze_all=True, name=''):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    if freeze_all:
        for p in model.backbone.parameters(): p.requires_grad=False
    else:
        for layer in [model.backbone.body.layer1, model.backbone.body.layer2]:
            for p in layer.parameters(): p.requires_grad=False
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model.to(device)

# ===================================================================
# LOAD ALL 3 SAVED MODELS
# ===================================================================
print('\nLoading saved model weights...')

# Check all model files exist
for fname in ['kitti_model_A.pth','kitti_model_B.pth','kitti_model_C.pth']:
    if not os.path.exists(fname):
        print(f'ERROR: {fname} not found in current folder.')
        print('Make sure you run this script from the same folder as the .pth files.')
        sys.exit(1)

model_A = build_model(NUM_CLASSES, freeze_all=True,  name='Exp A')
model_A.load_state_dict(torch.load('kitti_model_A.pth', map_location=device))
model_A.eval()
print('  Loaded: kitti_model_A.pth')

model_B = build_model(NUM_CLASSES, freeze_all=False, name='Exp B')
model_B.load_state_dict(torch.load('kitti_model_B.pth', map_location=device))
model_B.eval()
print('  Loaded: kitti_model_B.pth')

model_C = build_model(NUM_CLASSES, freeze_all=False, name='Exp C')
model_C.load_state_dict(torch.load('kitti_model_C.pth', map_location=device))
model_C.eval()
print('  Loaded: kitti_model_C.pth')

# ===================================================================
# EVALUATION FUNCTION (fixed np.trapezoid)
# ===================================================================
def compute_iou(b1,b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0.0

def evaluate(model, loader, name='Model'):
    model.eval()
    cls_ious=defaultdict(list)
    cls_dets=defaultdict(list)
    cls_gt=defaultdict(int)
    with torch.no_grad():
        for images,targets in loader:
            images=[img.to(device) for img in images]
            preds=model(images)
            for pred,target in zip(preds,targets):
                gt_boxes=target['boxes'].numpy()
                gt_labels=target['labels'].numpy()
                pb=pred['boxes'].cpu().numpy()
                pl=pred['labels'].cpu().numpy()
                ps=pred['scores'].cpu().numpy()
                mask=ps>=SCORE_THR
                pb,pl,ps=pb[mask],pl[mask],ps[mask]
                for lbl in gt_labels: cls_gt[int(lbl)]+=1
                gt_matched=np.zeros(len(gt_boxes),bool)
                for i in np.argsort(-ps):
                    best_iou,best_j=0.0,-1
                    for j,(gb,gl) in enumerate(zip(gt_boxes,gt_labels)):
                        if gt_matched[j] or int(gl)!=int(pl[i]): continue
                        iou=compute_iou(pb[i],gb)
                        if iou>best_iou: best_iou=iou; best_j=j
                    if best_iou>=IOU_THR and best_j>=0:
                        gt_matched[best_j]=True
                        cls_ious[int(pl[i])].append(best_iou)
                        cls_dets[int(pl[i])].append((ps[i],1))
                    else:
                        cls_dets[int(pl[i])].append((ps[i],0))

    print(f'\n  --- {name} @ IoU={IOU_THR} score_thr={SCORE_THR} ---')
    print(f'  {"Class":<15}{"mIoU":<10}{"AP":<10}{"GT count"}')
    print(f'  {"-"*40}')
    aps=[]
    for cid,cname in [(1,'Car'),(2,'Pedestrian'),(3,'Cyclist')]:
        ious=cls_ious[cid]; miou=np.mean(ious) if ious else 0.0
        dets=cls_dets[cid]; ngt=cls_gt[cid]
        if dets and ngt>0:
            ds=sorted(dets,key=lambda x:-x[0])
            tps=np.cumsum([d[1] for d in ds])
            fps=np.cumsum([1-d[1] for d in ds])
            prec=tps/(tps+fps); rec=tps/ngt
            # Fixed: use np.trapezoid (works in NumPy >= 2.0)
            try:
                ap=float(np.trapezoid(prec,rec)) if len(rec)>1 else 0.0
            except AttributeError:
                ap=float(np.trapz(prec,rec)) if len(rec)>1 else 0.0
        else: ap=0.0
        aps.append(ap)
        print(f'  {cname:<15}{miou:<10.4f}{ap:<10.4f}{ngt}')
    mAP=float(np.mean(aps))
    print(f'\n  mAP@{IOU_THR}: {mAP:.4f}')
    return mAP,aps

# ===================================================================
# RUN EVALUATION
# ===================================================================
print('\n' + '='*60)
print('EVALUATION — IoU & mAP on Test Set (Video16)')
print('='*60)

mAP_A, aps_A = evaluate(model_A, test_loader,   name='Exp A: Full Freeze')
mAP_B, aps_B = evaluate(model_B, test_loader,   name='Exp B: Partial Freeze')
mAP_C, aps_C = evaluate(model_C, test_loader_C, name='Exp C: Temporal T=3')

# ===================================================================
# PLOT mAP COMPARISON
# ===================================================================
print('\nGenerating mAP comparison chart...')
cls_names=['Car','Pedestrian','Cyclist']
x=np.arange(len(cls_names)); w=0.25

fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle('KITTI Object Detection — Evaluation Results (mAP@0.5)',
             fontsize=13,fontweight='bold')

axes[0].bar(x-w, aps_A, w, label='Exp A Full Freeze',   color='steelblue',     alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].bar(x,   aps_B, w, label='Exp B Partial Freeze', color='coral',          alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].bar(x+w, aps_C, w, label='Exp C Temporal T=3',  color='mediumseagreen', alpha=0.85, edgecolor='black', linewidth=0.5)

for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.3f', fontsize=8, fontweight='bold', padding=2)

axes[0].set_xticks(x); axes[0].set_xticklabels(cls_names, fontsize=11)
axes[0].set_ylabel('Average Precision (AP)', fontsize=11)
axes[0].set_title('AP per Class — All Experiments', fontsize=11)
axes[0].legend(fontsize=9); axes[0].set_ylim(0,1); axes[0].grid(axis='y',alpha=0.3)

best_mAP = max(mAP_A, mAP_B, mAP_C)
bars2 = axes[1].bar(
    ['Exp A\nFull Freeze','Exp B\nPartial Freeze','Exp C\nTemporal T=3'],
    [mAP_A,mAP_B,mAP_C],
    color=['steelblue','coral','mediumseagreen'],
    alpha=0.85, width=0.4, edgecolor='black', linewidth=0.5)
for bar,v in zip(bars2,[mAP_A,mAP_B,mAP_C]):
    axes[1].text(bar.get_x()+bar.get_width()/2., v+0.008,
                 f'{v:.4f}' + (' ★' if v==best_mAP else ''),
                 ha='center', fontsize=12, fontweight='bold',
                 color='darkgreen' if v==best_mAP else 'black')
axes[1].set_ylabel('mAP@0.5', fontsize=11)
axes[1].set_title('Mean Average Precision Comparison', fontsize=11)
axes[1].set_ylim(0,1); axes[1].grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.savefig('kitti_map_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: kitti_map_comparison.png')

# ===================================================================
# VISUALISE PREDICTIONS (BEST MODEL)
# ===================================================================
print('\nGenerating prediction visualisation...')

best_idx  = [mAP_A,mAP_B,mAP_C].index(best_mAP)
best_name = ['Exp A: Full Freeze','Exp B: Partial Freeze','Exp C: Temporal T=3'][best_idx]
best_model= [model_A,model_B,model_C][best_idx]
best_test = [test_dataset,test_dataset,test_temp][best_idx]
best_loader=[test_loader,test_loader,test_loader_C][best_idx]

idx_gt = [i for i in range(len(best_test))
          if len(best_test.samples[i]['annots'].get(
             best_test.samples[i]['frame_idx'],[]))>0]

if not idx_gt:
    print('  No annotated frames found in test set.')
else:
    chosen = random.sample(idx_gt, min(6, len(idx_gt)))
    fig,axes = plt.subplots(2,3,figsize=(18,9))
    fig.suptitle(f'Predictions vs Ground-Truth — {best_name}\n'
                 f'Test Sequence: Video16 | White dashed=GT | Coloured=Predictions',
                 fontsize=13,fontweight='bold')

    for ax,idx in zip(axes.flat,chosen):
        img_t,target = best_test[idx]
        ax.imshow(img_t.permute(1,2,0).numpy())

        # Ground truth — dashed white
        for box,lbl in zip(target['boxes'],target['labels']):
            x1,y1,x2,y2=box.tolist()
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                linewidth=2,edgecolor='white',facecolor='none',linestyle='--'))
            ax.text(x1,y2+2,f'GT:{ID2NAME.get(lbl.item(),"?")}',
                color='white',fontsize=7,
                bbox=dict(facecolor='black',alpha=0.4,pad=1,edgecolor='none'))

        # Predictions — solid coloured
        with torch.no_grad():
            pred = best_model([img_t.to(device)])[0]
        pb=pred['boxes'].cpu().numpy()
        pl=pred['labels'].cpu().numpy()
        ps=pred['scores'].cpu().numpy()
        n_pred=0
        for b,l,s in zip(pb,pl,ps):
            if s<SCORE_THR: continue
            n_pred+=1; x1,y1,x2,y2=b
            color=PRED_COLORS.get(int(l),'yellow')
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                linewidth=2,edgecolor=color,facecolor='none'))
            ax.text(x1,y1-3,f'{ID2NAME.get(int(l),"?")} {s:.2f}',
                color=color,fontsize=7,fontweight='bold',
                bbox=dict(facecolor='black',alpha=0.5,pad=1,edgecolor='none'))

        ax.set_title(f'GT:{len(target["boxes"])} Pred:{n_pred}',fontsize=9)
        ax.axis('off')

    legend_els=[
        Line2D([0],[0],color='white',linewidth=2,linestyle='--',label='Ground-Truth'),
        Line2D([0],[0],color='red',linewidth=2,label='Car'),
        Line2D([0],[0],color='deepskyblue',linewidth=2,label='Pedestrian'),
        Line2D([0],[0],color='lime',linewidth=2,label='Cyclist')]
    fig.legend(handles=legend_els,loc='lower center',ncol=4,fontsize=11)
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig('kitti_predictions.png',dpi=150,bbox_inches='tight')
    plt.close()
    print('  Saved: kitti_predictions.png')

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print('\n' + '='*65)
print('FINAL RESULTS — CET3013 Task 2: KITTI Object Detection')
print('='*65)
print(f'{"Experiment":<28}{"Car AP":<10}{"Ped AP":<10}{"Cyc AP":<10}{"mAP@0.5"}')
print('-'*65)
for name,aps,mAP in [
    ('Exp A: Full Freeze',   aps_A, mAP_A),
    ('Exp B: Partial Freeze',aps_B, mAP_B),
    ('Exp C: Temporal T=3',  aps_C, mAP_C)]:
    mark=' <- BEST' if mAP==best_mAP else ''
    print(f'{name:<28}{aps[0]:<10.4f}{aps[1]:<10.4f}{aps[2]:<10.4f}{mAP:.4f}{mark}')

print(f'\nBest model: {best_name}')
print(f'\nOutput files saved:')
for f in ['kitti_map_comparison.png','kitti_predictions.png']:
    status='OK' if os.path.exists(f) else 'MISSING'
    print(f'  [{status}] {f}')

print('\nDone! Use these results to complete Table 2 in your report.')
