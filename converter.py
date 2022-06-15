import torch
import torch_tensorrt
import torchvision.models as models

resnet101 = models.resnet101(pretrained=True)
resnet101 = resnet101.eval()

inputs_fp32  = [torch_tensorrt.Input([1,3,224,224],dtype=torch.float)]
enabled_precisions_fp32 = {torch.float}
trt_ts_module_fp32 = torch_tensorrt.compile(resnet101,inputs=inputs_fp32,enabled_precisions=enabled_precisions_fp32,require_full_compilation=True)
torch.jit.save(trt_ts_module_fp32,"resnet101_fp32.ts")
del trt_ts_module_fp32


inputs_fp16  = [torch_tensorrt.Input([1,3,224,224],dtype=torch.half)]
enabled_precisions_fp16 = {torch.half}
trt_ts_module_fp16 = torch_tensorrt.compile(resnet101,inputs=inputs_fp16,enabled_precisions=enabled_precisions_fp16,require_full_compilation=True)
torch.jit.save(trt_ts_module_fp16,"resnet101_fp16.ts")
del trt_ts_module_fp16

