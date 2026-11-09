"""Hash deterministic artificial inputs and valid frozen fusion outputs only."""
import argparse,hashlib,json
import numpy as np
from paper2_runtime.preprocessing import cnn_iq8_tensor,deit_png_tensor,reference_roi_tensor,reference_cached_tensor
from paper2_runtime.fusion import common5_fusion,matched_common5_fusion,source94_fusion,source94_reference_average

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
config=json.load(open(args.config,encoding='utf-8'));rng=np.random.default_rng(4281);out={}
def add(name,value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    a=np.ascontiguousarray(value);out[name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
for index,shape in enumerate([(1,1),(3,7),(8,8),(51,73),(200,300)]):
    image=rng.integers(0,256,size=shape,dtype=np.uint8)
    add('cnn_'+str(index),cnn_iq8_tensor(image));add('deit_'+str(index),deit_png_tensor(image))
    add('reference_'+str(index),reference_roi_tensor(image,[0,0,shape[1],shape[0]]))
add('reference_cached',reference_cached_tensor(rng.integers(0,256,size=(224,224),dtype=np.uint8)))
def probs():
    x=rng.random((7,4),dtype=np.float32);return x/x.sum(1,keepdims=True)
image,deit=probs(),probs()
for fold in range(5):
    for strategy in ['R10','R11']:add('common_'+strategy+'_'+str(fold),common5_fusion(image,deit,fold,strategy,config))
    for arm in config['matched_common5']['selected_image_weights']:
        for strategy in ['fixed','selected']:add('matched_'+arm+'_'+strategy+'_'+str(fold),matched_common5_fusion(image,deit,fold,arm,strategy,config))
source_c=[probs() for _ in range(5)];source_d=[probs() for _ in range(5)]
add('source94',source94_fusion(source_c,source_d,config));add('sr1',source94_reference_average(source_c))
print(json.dumps(out,sort_keys=True))
