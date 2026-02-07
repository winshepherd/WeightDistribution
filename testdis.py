import matplotlib.pyplot as plt
import numpy as np
from unsloth import FastLanguageModel
import torch
from scipy import spatial
import seaborn as sns


def calc_svd(tensor):
    t = tensor.float()
    _,s,_ = torch.linalg.svd(t, full_matrices=False)
    ts = s.detach().numpy()
    return ts

def get_vecdis(ts1,ts2,k):
    dis = spatial.distance.cosine(ts1[:k], ts2[:k])
    # U, s, V = torch.linalg.svd(tensor1, full_matrices=False)
    # m =  U[:, :k] @ torch.diag_embed(s[:k]) @ V[:k, :]  
    return round(dis,8)


def getdistance(m_name,disfname,rank=16,layernum = 16):
    max_seq_length = 2048
    # dtype = None # Auto detection of dtype
    model, _ = FastLanguageModel.from_pretrained(
    model_name=m_name,
    # model_name = "/src/models/smo135/",
    # model_name = "./outputs_ms/",
    load_in_4bit = False,
    max_seq_length = max_seq_length,
    # dtype =  torch.bfloat16
    )
    # model = FastLanguageModel.get_peft_model(
    # model,
    # r = 16, # LoRA rank
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # lora_alpha = 16,
    # lora_dropout = 0, # Optimized at 0
    # bias = "none", # No additional bias terms
    # use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
    # random_state = 3407,
    # use_rslora = False, # Rank stabilized LoRA, can be enabled for stability
    # )
    # loraa = model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight
    # loraa = (model.model.layers[0].self_attn.q_proj.weight).cpu()
    # loraa = (model.model.layers[0].self_attn.q_proj.weight).cpu()    
    # lorab = (model.model.layers[0].mlp.down_proj.weight).cpu()  
    w_q = []
    w_k = []
    w_v = []
    w_o = []
    w_g = []
    w_up = []
    w_down = []
    w_arry = []
    
    netshape = 7*layernum
    for i in range(layernum):
        w_q.append(calc_svd((model.model.layers[i].self_attn.q_proj.weight).cpu()))
        w_k.append(calc_svd((model.model.layers[i].self_attn.k_proj.weight).cpu()))
        w_v.append(calc_svd((model.model.layers[i].self_attn.v_proj.weight).cpu()))
        w_o.append(calc_svd((model.model.layers[i].self_attn.o_proj.weight).cpu()))
        w_g.append(calc_svd((model.model.layers[i].mlp.gate_proj.weight).cpu()))
        w_up.append(calc_svd((model.model.layers[i].mlp.up_proj.weight).cpu()))
        w_down.append(calc_svd((model.model.layers[i].mlp.down_proj.weight).cpu()))
    w_arry.append(w_q)
    w_arry.append(w_k)
    w_arry.append(w_v) 
    w_arry.append(w_o)
    w_arry.append(w_g)
    w_arry.append(w_up)
    w_arry.append(w_down)
    dismatrix = np.zeros((netshape,netshape))
    for i in range(7):
        for j in range(layernum):
            v1 = w_arry[i][j]
            for m in range(7):
                for n in range(layernum):
                    v2 = w_arry[m][n]
                    if i == m and j == n:
                        continue
                    else:
                        dis = get_vecdis(v1,v2,rank)
                        dismatrix[i*layernum+j][m*layernum+n] = dis
                        dismatrix[m*layernum+n][i*layernum+j] = dis
    np.savetxt(X=dismatrix,fname=disfname)
    return dismatrix

def getdistance2(m_name,disfname,rank=16,layernum = 16):
    max_seq_length = 2048
    model, _ = FastLanguageModel.from_pretrained(
    model_name=m_name,

    load_in_4bit = False,
    max_seq_length = max_seq_length,
    # dtype =  torch.bfloat16
    )
    # model = FastLanguageModel.get_peft_model(
    # model,
    # r = 16, # LoRA rank
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # lora_alpha = 16,
    # lora_dropout = 0, # Optimized at 0
    # bias = "none", # No additional bias terms
    # use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
    # random_state = 3407,
    # use_rslora = False, # Rank stabilized LoRA, can be enabled for stability
    # )
    # loraa = model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight
    # loraa = (model.model.layers[0].self_attn.q_proj.weight).cpu()
    # loraa = (model.model.layers[0].self_attn.q_proj.weight).cpu()    
    # lorab = (model.model.layers[0].mlp.down_proj.weight).cpu()  
    w_arry = []
    
    netshape = 7*layernum
    for i in range(layernum):
        w_arry.append(calc_svd((model.model.layers[i].self_attn.q_proj.weight).cpu()))
        w_arry.append(calc_svd((model.model.layers[i].self_attn.k_proj.weight).cpu()))
        w_arry.append(calc_svd((model.model.layers[i].self_attn.v_proj.weight).cpu()))
        w_arry.append(calc_svd((model.model.layers[i].self_attn.o_proj.weight).cpu()))
        w_arry.append(calc_svd((model.model.layers[i].mlp.gate_proj.weight).cpu()))
        w_arry.append(calc_svd((model.model.layers[i].mlp.up_proj.weight).cpu()))
        w_arry.append(calc_svd((model.model.layers[i].mlp.down_proj.weight).cpu()))
    dismatrix = np.zeros((netshape,netshape))
    for i in range(len(w_arry)):
        v1 = w_arry[i]
        for j in range(i,len(w_arry)):
            v2 = w_arry[j]
            if i == j:
                continue
            else:
                dis = get_vecdis(v1,v2,rank)    
                dismatrix[i][j] = dis
                dismatrix[j][i] = dis                
    np.savetxt(X=dismatrix,fname=disfname)
    return dismatrix



def getcos(tensor1,tensor2,k):
    t1 = tensor1.float()
    t2 = tensor2.float()
    _,s1,_ = torch.linalg.svd(t1, full_matrices=False)
    _,s2,_ = torch.linalg.svd(t2, full_matrices=False)
    ts1 = s1.detach().numpy()
    ts2 = s2.detach().numpy()
    dis = spatial.distance.cosine(ts1[:k], ts2[:k])
    # U, s, V = torch.linalg.svd(tensor1, full_matrices=False)
    # m =  U[:, :k] @ torch.diag_embed(s[:k]) @ V[:k, :]  
    return round(dis,8)

def plotheat():
    data = np.loadtxt('dis.txt',dtype=np.float32, delimiter=' ')
    data = data * 10
    ax = sns.heatmap(data)
    # plt.show()
    plt.savefig('./img/dis.png')

def plotpareto(fname):
    data = np.loadtxt(fname)
    s=[]
    for i in range(112,140):
        for j in range(140,168):
            if data[i][j] > 0:
                s.append(data[i][j])
    counts, bins, _ =plt.hist(s, bins=20)
    frequencies = counts / len(s)
    plt.clf()
    plt.bar(bins[:-1], frequencies, width=np.diff(bins), edgecolor='black', align='edge')
    plt.title('MLP-MLP')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    plt.ylim(0, 1)  # 设置y轴范围为0到1
    plt.show()

def distinct(fname):
    data = np.loadtxt(fname)
    unique_data = np.unique(data)
    print(len(unique_data))

# distinct("./data/q7b_16.txt")
d = getdistance("/src/models/qwen7b/","./data/qwen7b_256.txt",rank=256,layernum=28)
# print(d)
# plotheat()
# plotpareto('./data/q7b_16.txt')
# # s = getweight2("./outputs_ms/",)
# s = getweight()
# counts, bins, _ =plt.hist(s, bins=20)
# frequencies = counts / len(s)
# plt.clf()
# plt.bar(bins[:-1], frequencies, width=np.diff(bins), edgecolor='black', align='edge')
# # plt.title('Manually Normalized Histogram - how2matplotlib.com')
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# plt.ylim(0, 1)  # 设置y轴范围为0到1
# # plt.show()
# plt.savefig('./img/dis_orgo.png')

