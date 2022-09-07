#这是一个集成了多种动漫超分辨率算法的脚本
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
from vsmlrt import CUGAN,Backend#vs-mlrt
from upcunet_v20220227_vs import RealWaifuUpScaler#pytorch

core = vs.core
core.num_threads = 16
core.max_cache_size = 8000 

#pytorch
upscaler=RealWaifuUpScaler(2,r"D:\VS_test\Real_cugan_pth\pro-conservative-up2x.pth",False,"cuda:0",3,0,1)

#vs-mlrt
def upscale(clip):
        clip = CUGAN(clip, noise=0, scale=2, tiles=4,version=2,alpha=1, backend=Backend.ORT_CUDA())
        #clip = CUGAN(clip, noise=0,scale=2,tiles=4,version=1,backend=Backend.ORT_CPU())
        return clip
       
#src是视频源文件，res是中间处理文件,这里区分以便于后面对比预览
src = core.lsmas.LWLibavSource(r"D:\VS_test\NCED1.m2ts")


src = core.resize.Bicubic(clip=src,format=vs.YUV444P16)#转换为Yuv444 16bit,这里是参照组，最好不要修改。添加matrix_s参数可以指定色域。
res = src#中间处理文件


#滤镜预处理分界线-------------------------------------------------------------滤镜预处理分界线

#滤镜这一块的使用方式可以根据需求自行调整参数和顺序，也可以增删新的滤镜函数
#如果你不理解这些滤镜的用途和用法，建议可以先查看文档再回来调试

#res = core.std.CropRel(res, 0, 0, 22, 20)#切黑边

#res = haf.QTGMC(res, Preset='Slower', TFF=True, FPSDivisor=2)#反交错。这里为了通用性使用了预设，参数可以查文档自定义。

#res  = core.neo_f3kdb.Deband(res,preset="medium",output_depth=16)#去色带。这里为了通用性使用了预设，参数可以查文档自定义。

#res = rektlvls(res,rownum=[1067,1068,1070,1071,1072,1073,1074,1075,1078], rowval=[-2,-4,4,0,-3,-10,-20,-10,15])
#去脏边，目前我只知道通过手动指定调整。如果有更好的方法可以相互交流改进。

#res=core.cas.CAS(res, sharpness=1)#自适应锐化。这里为了通用性使用了简单的参数，其他参数可以查文档自定义

#res=core.knlm.KNLMeansCL(res,d=1,a=2,h=3,device_type='GPU',device_id=0)#去噪点。这里为了通用性使用了简单的参数，其他参数可以查文档自定义

#res = core.resize.Bicubic(clip=res,format=vs.YUV444P16)#这一步可以指定长宽width,height来进行缩放,添加matrix_s参数可以指定色域。

#滤镜预处理分界线-------------------------------------------------------------滤镜预处理分界线



#超分辨率处理|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||超分辨率处理#


res = mvf.ToRGB(res,matrix="709",depth=32)#添加matrix参数可以指定色域。real-cugan超分辨率前必须转为RGB格式。

#Real-cugan超分辨率,上面定义的pytorch和vs-mlrt函数
res = upscale(res) #vs-mlrt,推荐使用
#res = upscaler(res) #pytorch

#Anime4K超分辨率
#res=core.anime4kcpp.Anime4KCPP(res, GPUMode = 1, ACNet = 1, zoomFactor = 2, HDN = 1, HDNLevel = 1)


#经过测试下面两个函数按相同的参数转换出来的单帧图片是一致的，保存为PNG大小一致。想用哪个用哪个
#src = mvf.ToYUV(src,depth=16,matrix="709",css="444")

res = core.resize.Bicubic(clip=res,matrix_s="709",format=vs.YUV444P16)#添加matrix_s参数可以指定色域,这一步可以指定长宽width,height来进行缩放

#超分辨率处理|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||超分辨率处理#



#最后滤镜处理和输出================================================================================最后滤镜处理和输出#

#res = adptvgrnMod(res, size=3, strength=[10,10], sharp=33, luma_scaling=50, seed=3, show_mask=0)#全局动态噪点，参数查文档




# (很重要)当Debug=1的时候,交错显示视频源和超分辨率处理后的视频，使用Debug=1时请保证输入的cilp的width和height相同
# (很重要)当Debug=2的时候,将源视频画面和超分后的视频画面左右拼接为一个画面，使用Debug=2时请保证输入的cilp的width和height相同
# (很重要)当Debug=其他值的时候,即为成片预览，可以使用vspipe和ffmpeg输出成品
Debug = 0
if Debug == 1: 
    src=core.resize.Bicubic(clip=src,width=3840,height=2160,format=vs.YUV444P16)#需要保证二者格式和长宽一致,这里使用bicubic放大源
    res=core.resize.Bicubic(clip=res,width=3840,height=2160,format=vs.YUV444P16)#需要保证二者格式和长宽一致
    compare=core.std.Interleave([src,res],modify_duration=False)
    compare=mvf.ToRGB(compare,full=False,depth=16).set_output()#交错预览，前一帧是源后一帧是超分后的结果，交错显示，需要保证二者格式和长宽一致。
elif Debug == 2:
    src=core.resize.Bicubic(clip=src,width=3840,height=2160,format=vs.YUV444P16)#需要保证二者格式和长宽一致,这里使用bicubic放大源
    res=core.resize.Bicubic(clip=res,width=3840,height=2160,format=vs.YUV444P16)#需要保证二者格式和长宽一致
    compare=lvsfunc.comparison.tile(src,res)#预览左边是源，右边是处理后的结果
    compare.set_output()
else:
    res.set_output()
#最后处理和输出================================================================================最后处理和输出#
