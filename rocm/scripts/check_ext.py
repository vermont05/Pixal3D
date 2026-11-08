# Checks the HIP builds: imports, FlexGEMM grid_sample_3d vs torch grid_sample, CuMesh basic ops.
import inspect, torch, cumesh, flex_gemm, o_voxel
from flex_gemm.ops.grid_sample import grid_sample_3d
import torch.nn.functional as F
print("imports OK", torch.cuda.get_device_name(0)); dev="cuda"; torch.manual_seed(0)
print(inspect.signature(grid_sample_3d))
# smooth dense volume stored sparsely: every voxel present, so the result must equal dense trilinear sampling
for R,C in [(16,3),(32,6),(64,6)]:
    g=torch.stack(torch.meshgrid(*[torch.arange(R,device=dev)]*3,indexing="ij"),-1).float()
    vol=torch.stack([torch.sin(g[...,0]*.3+k)+torch.cos(g[...,1]*.2)*g[...,2]/R for k in range(C)],-1)  # [R,R,R,C]
    coords=g.long().reshape(-1,3); feats=vol.reshape(-1,C)
    n=20000; q=(0.5+torch.rand(n,3,device=dev)*(R-1))   # voxel-index space with voxel centres at +0.5, interior only
    out=grid_sample_3d(feats,torch.cat([torch.zeros_like(coords[:,:1]),coords],-1),shape=torch.Size([1,C,R,R,R]),grid=q.reshape(1,-1,3),mode="trilinear")
    # reference: torch grid_sample on [1,C,D,H,W]; grid xyz maps x->W(last), so permute the volume axes
    d=vol.permute(3,2,1,0)[None]
    gn=(q/R*2-1).reshape(1,1,1,-1,3)
    ref=F.grid_sample(d,gn,mode="bilinear",align_corners=False)[0,:,0,0].T
    out=out.reshape(ref.shape); err=(out-ref).abs()
    print(f"grid_sample_3d R={R} C={C}: max err {err.max():.3e} mean {err.mean():.3e} (signal rms {ref.std():.3f})")
# CuMesh: unit cube, closed mesh with known area and BVH distances
v=torch.tensor([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)],device=dev)
f=torch.tensor([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],device=dev,dtype=torch.int32)
m=cumesh.CuMesh(); m.init(v,f); print("CuMesh verts/faces",m.num_vertices,m.num_faces)
bvh=cumesh.cuBVH(v,f); p=torch.tensor([[0.5,0.5,2.0],[0.5,0.5,0.5],[3.,0.5,0.5]],device=dev)
d,fid,uvw=bvh.unsigned_distance(p,return_uvw=True); print("BVH distances (expect 1, .5, 2):",d.tolist()," uvw sums",uvw.sum(1).tolist())
