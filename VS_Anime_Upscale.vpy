#这是一个集成了多种动漫超分辨率算法的脚本,拥有详细的注释，如果不想看注释的可以看另一个简化版的文件。
#增删或修改里面的函数以及参数的设置请确保你理解它们的用途并且清楚它们的用法。
#脚本尚处于测试阶段，有问题请联系NangInShell@https://github.com/NangInShell
#欢迎大佬们提BUG和改进建议


import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import lvsfunc

from rekt import *
from adptvgrnMod import adptvgrnMod
from vsmlrt import CUGAN,RealESRGAN,Waifu2x,Backend#vs-mlrt

core = vs.core
core.num_threads = 16
core.max_cache_size = 8000 

#vs-mlrt
def waifu2x_US(clip):
        clip=Waifu2x(clip,noise=-1,scale=2,tiles=2,tilesize=None,overlap=None,model=1,backend=Backend.ORT_CUDA(),preprocess=True)
        return clip

def cugan_US(clip):
        clip = CUGAN(clip, noise=0, scale=2, tiles=2,version=2,alpha=1, backend=Backend.ORT_CUDA())
        return clip
       
def esrgan_US(clip):
        clip = RealESRGAN(clip, scale=2,tiles=2, tilesize=None, overlap=None,model=2, backend=Backend.ORT_CUDA())
        return clip
       
#src是视频源文件，res是中间处理文件,这里区分以便于后面对比预览
src = core.lsmas.LWLibavSource(r"D:\VS_NangInShell\VS_Nang\Test.png")
src = core.resize.Bicubic(clip=src,format=vs.YUV420P10)#转换为Yuv420 10bit,这里是参照组，最好不要修改。添加matrix_s参数可以指定色域。
res = src#中间处理文件

#滤镜预处理分界线-------------------------------------------------------------滤镜预处理分界线

#滤镜这一块的使用方式可以根据需求自行调整参数和顺序，也可以增删新的滤镜函数
#如果你不理解这些滤镜的用途和用法，建议可以先查看文档再回来调试

#res = core.std.CropRel(res, 0, 0, 22, 20)#切黑边

#res = taa.TAAmbk(res, aatype=-3, preaa=-1, strength=0.1, mtype=2, aapair=1, cycle=5, sharp=0)#抗锯齿

#res = haf.QTGMC(res, Preset='Slower', TFF=True, FPSDivisor=2)#反交错。这里为了通用性使用了预设，参数可以查文档自定义。

#res  = core.neo_f3kdb.Deband(res,preset="medium",output_depth=16)#去色带。这里为了通用性使用了预设，参数可以查文档自定义。

#res = rektlvls(res,rownum=[1067,1068,1070,1071,1072,1073,1074,1075,1078], rowval=[-2,-4,4,0,-3,-10,-20,-10,15])
#去脏边，目前我只知道通过手动指定调整。如果有更好的方法可以相互交流改进。

#res=core.cas.CAS(res, sharpness=1)#自适应锐化。这里为了通用性使用了简单的参数，其他参数可以查文档自定义

#res=core.knlm.KNLMeansCL(res,d=1,a=2,h=3,device_type='GPU',device_id=0)#去噪点。这里为了通用性使用了简单的参数，其他参数可以查文档自定义

#滤镜预处理分界线-------------------------------------------------------------滤镜预处理分界线



#超分辨率处理|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||超分辨率处理#

#res = core.resize.Bicubic(clip=res)#这一步可以指定长宽width,height来进行缩放,添加matrix_s参数可以指定色域。.

res = core.resize.Bicubic(clip=res,range=0,format=vs.RGB48)#添加matrix参数可以指定色域。real-cugan超分辨率前必须转为RGB格式。
res=core.fmtc.bitdepth(res, bits=32)

#超分辨率函数,上面定义的函数
#res = waifu2x_US(res) 
#res = cugan_US(res)
#res = esrgan_US(res) 
#res=core.anime4kcpp.Anime4KCPP(res, GPUMode = 1, ACNet = 1, zoomFactor = 2, HDN = 1, HDNLevel = 1)


res = core.resize.Bicubic(clip=res,matrix_s='709',format=vs.YUV420P10)#添加matrix_s参数可以指定色域,这一步可以指定长宽width,height来进行缩放

#超分辨率处理|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||超分辨率处理#



#最后滤镜处理和输出================================================================================最后滤镜处理和输出#

#res = adptvgrnMod(res, size=1, strength=[500,500], sharp=100, fade_edges=10, luma_scaling=50, seed=1, show_mask=0)#全局动态噪点，参数查文档

# Debug=1，片源和成品交错显示，可以在预览通过键盘左右键对比；Debug=0，正常输出。
# (很重要)当Debug=1的时候,交错显示视频源和超分辨率处理后的视频，使用Debug=1时请保证输入的cilp的width和height相同
# (很重要)当Debug=2的时候,将源视频画面和超分后的视频画面左右拼接为一个画面，使用Debug=2时请保证输入的cilp的width和height相同
# (很重要)当Debug=其他值的时候,即为成片预览，可以使用vspipe和ffmpeg输出成品
Debug = 0
if Debug == 1: 
    src=core.resize.Bicubic(clip=src,width=1920,height=1080,format=vs.YUV444P16)#需要保证二者格式和长宽一致,这里使用bicubic放大源
    res=core.resize.Bicubic(clip=res,width=1920,height=1080,format=vs.YUV444P16)#需要保证二者格式和长宽一致
    compare=core.std.Interleave([src,res],modify_duration=False)
    compare=mvf.ToRGB(compare,full=False,depth=16).set_output()#交错预览，前一帧是源后一帧是超分后的结果，交错显示，需要保证二者格式和长宽一致。
elif Debug == 2:
    src=core.resize.Bicubic(clip=src,width=1920,height=1080,format=vs.YUV444P16)#需要保证二者格式和长宽一致,这里使用bicubic放大源
    res=core.resize.Bicubic(clip=res,width=1920,height=1080,format=vs.YUV444P16)#需要保证二者格式和长宽一致
    compare=lvsfunc.comparison.tile(src,res)#预览左边是源，右边是处理后的结果
    compare.set_output()
else:
    res.set_output()
#最后处理和输出================================================================================最后处理和输出#
