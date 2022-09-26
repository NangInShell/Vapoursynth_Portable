import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import lvsfunc

from rekt import *
from adptvgrnMod import adptvgrnMod
from vsmlrt import CUGAN,RealESRGAN,Waifu2x,Backend

core = vs.core
core.num_threads = 16
core.max_cache_size = 8000 

def waifu2x_US(clip):
        clip=Waifu2x(clip,noise=-1,scale=2,tiles=2,tilesize=None,overlap=None,model=1,backend=Backend.ORT_CUDA(),preprocess=True)
        return clip

def cugan_US(clip):
        clip = CUGAN(clip, noise=0, scale=2, tiles=2,version=2,alpha=1, backend=Backend.ORT_CUDA())
        return clip
       
def esrgan_US(clip):
        clip = RealESRGAN(clip, scale=2,tiles=2, tilesize=None, overlap=None,model=2, backend=Backend.ORT_CUDA())
        return clip

src = core.lsmas.LWLibavSource(r"D:\VS_test\VS脚本\MV.mkv")
src = core.resize.Bicubic(clip=src,width=1920,height=1080,format=vs.YUV420P10)
res = src

#滤镜预处理分界线-------------------------------------------------------------滤镜预处理分界线

#res = taa.TAAmbk(res, aatype=-3, preaa=-1, strength=0.1, mtype=2, aapair=1, cycle=5, sharp=0)

#res = core.std.CropRel(res, 0, 0, 22, 20)

#res = haf.QTGMC(res, Preset='Slower', TFF=True, FPSDivisor=2)

#res  = core.neo_f3kdb.Deband(res,preset="medium",output_depth=16)

#res = rektlvls(res,rownum=[1067,1068,1070,1071,1072,1073,1074,1075,1078], rowval=[-2,-4,4,0,-3,-10,-20,-10,15])

#res=core.cas.CAS(res, sharpness=1)

#res=core.knlm.KNLMeansCL(res,d=1,a=2,h=3,device_type='GPU',device_id=0)

#res = core.resize.Bicubic(clip=res,width=1920,height=1080)

#滤镜预处理分界线-------------------------------------------------------------滤镜预处理分界线



#超分辨率处理|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||超分辨率处理#

res = mvf.ToRGB(res,matrix="709",depth=32)

#res = waifu2x_US(res) 
res = cugan_US(res)
#res = esrgan_US(res) 
#res=core.anime4kcpp.Anime4KCPP(res, GPUMode = 1, ACNet = 1, zoomFactor = 2, HDN = 1, HDNLevel = 1)

res = core.resize.Bicubic(clip=res,matrix_s='709',format=vs.YUV420P10)

#超分辨率处理|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||超分辨率处理#

#res = adptvgrnMod(res, size=1, strength=[500,500], sharp=100, fade_edges=10, luma_scaling=50, seed=1, show_mask=0)

#最后滤镜处理和输出================================================================================最后滤镜处理和输出#

Debug = 0
if Debug == 1: 
    src=core.resize.Bicubic(clip=src,width=1920,height=1080,format=vs.YUV444P16)
    res=core.resize.Bicubic(clip=res,width=1920,height=1080,format=vs.YUV444P16)
    compare=core.std.Interleave([src,res],modify_duration=False)
    compare=mvf.ToRGB(compare,full=False,depth=16).set_output()
elif Debug == 2:
    src=core.resize.Bicubic(clip=src,width=1920,height=1080,format=vs.YUV444P16)
    res=core.resize.Bicubic(clip=res,width=1920,height=1080,format=vs.YUV444P16)
    compare=lvsfunc.comparison.tile(src,res)
    compare.set_output()
else:
    res.set_output()
#最后处理和输出================================================================================最后处理和输出#