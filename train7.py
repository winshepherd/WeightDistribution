import torch
import time
from unsloth import FastLanguageModel,is_bfloat16_supported
from trl import SFTTrainer
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt,train_on_responses_only
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import copy
EOS_TOKEN = ''


def getUV(lora_model,i,layername,k,org_model):
    if layername == 'q_proj':
        tensor = org_model.model.model.layers[i].self_attn.q_proj.base_layer.weight
        a = lora_model.layers[i].self_attn.q_proj.lora_A.default.weight
        b = lora_model.layers[i].self_attn.q_proj.lora_B.default.weight
    elif layername == 'k_proj':
        tensor = org_model.model.model.layers[i].self_attn.k_proj.base_layer.weight
        a = lora_model.layers[i].self_attn.k_proj.lora_A.default.weight
        b = lora_model.layers[i].self_attn.k_proj.lora_B.default.weight
    elif layername == 'v_proj':
        tensor = org_model.model.model.layers[i].self_attn.v_proj.base_layer.weight
        a = lora_model.layers[i].self_attn.v_proj.lora_A.default.weight
        b = lora_model.layers[i].self_attn.v_proj.lora_B.default.weight
    elif layername == 'o_proj':
        tensor = org_model.model.model.layers[i].self_attn.o_proj.base_layer.weight
        a = lora_model.layers[i].self_attn.o_proj.lora_A.default.weight
        b = lora_model.layers[i].self_attn.o_proj.lora_B.default.weight
    elif layername == 'gate_proj':
        tensor = org_model.model.model.layers[i].mlp.gate_proj.base_layer.weight
        a = lora_model.layers[i].mlp.gate_proj.lora_A.default.weight
        b = lora_model.layers[i].mlp.gate_proj.lora_B.default.weight
    elif layername == 'up_proj':
        tensor = org_model.model.model.layers[i].mlp.up_proj.base_layer.weight
        a = lora_model.layers[i].mlp.up_proj.lora_A.default.weight
        b = lora_model.layers[i].mlp.up_proj.lora_B.default.weight
    elif layername == 'down_proj':
        tensor = org_model.model.model.layers[i].mlp.down_proj.base_layer.weight
        a = lora_model.layers[i].mlp.down_proj.lora_A.default.weight
        b = lora_model.layers[i].mlp.down_proj.lora_B.default.weight
    tensor1 = tensor.float()
    U, s, V = torch.linalg.svd(tensor1)
    s2 = s[:k].sqrt()
    # A = U[:, :k] @ torch.diag_embed(s2)
    # B = torch.diag_embed(s2) @ V[:k, :]    
    B = (U[:, :k] @ torch.diag_embed(s2))*0.3+b*0.7
    A = ((torch.diag_embed(s2) @ V[:k, :]))*0.3+a*0.7
    tensor0 = (tensor-(B @ A)).to(torch.bfloat16)
    return torch.nn.Parameter(A),torch.nn.Parameter(B),torch.nn.Parameter(tensor0)


def merge_model(lora_model,org_model,k):
    for i in range(16):
        lora_model.layers[i].self_attn.q_proj.lora_A.default.weight, lora_model.layers[i].self_attn.q_proj.lora_B.default.weight,lora_model.layers[i].self_attn.q_proj.base_layer.weight= getUV(lora_model,i,'q_proj',k,org_model)
        lora_model.layers[i].self_attn.k_proj.lora_A.default.weight, lora_model.layers[i].self_attn.k_proj.lora_B.default.weight,lora_model.layers[i].self_attn.k_proj.base_layer.weight= getUV(lora_model,i,'k_proj',k,org_model)
        lora_model.layers[i].self_attn.v_proj.lora_A.default.weight, lora_model.layers[i].self_attn.v_proj.lora_B.default.weight,lora_model.layers[i].self_attn.v_proj.base_layer.weight= getUV(lora_model,i,'v_proj',k,org_model)
        lora_model.layers[i].self_attn.o_proj.lora_A.default.weight, lora_model.layers[i].self_attn.o_proj.lora_B.default.weight,lora_model.layers[i].self_attn.o_proj.base_layer.weight= getUV(lora_model,i,'o_proj',k,org_model)
        lora_model.layers[i].mlp.gate_proj.lora_A.default.weight,lora_model.layers[i].mlp.gate_proj.lora_B.default.weight,lora_model.layers[i].mlp.gate_proj.base_layer.weight = getUV(lora_model,i,'gate_proj',k,org_model)
        lora_model.layers[i].mlp.up_proj.lora_A.default.weight,lora_model.layers[i].mlp.up_proj.lora_B.default.weight,lora_model.layers[i].mlp.up_proj.base_layer.weight = getUV(lora_model,i,'up_proj',k,org_model)
        lora_model.layers[i].mlp.down_proj.lora_A.default.weight,lora_model.layers[i].mlp.down_proj.lora_B.default.weight,lora_model.layers[i].mlp.down_proj.base_layer.weight = getUV(lora_model,i,'down_proj',k,org_model)
    return lora_model


def getmodel0():
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/src/models/llama32-1b",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Reduce memory usage with 4-bit quantization
    dtype =  torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized at 0
    bias = "none", # No additional bias terms
    use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
    random_state = 3407,
    use_rslora = False, # Rank stabilized LoRA, can be enabled for stability
    )
    return model

def getmodel():
    max_seq_length = 2048
    # dtype = None # Auto detection of dtype
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/src/models/llama32-1b",
    max_seq_length = max_seq_length,
    dtype =  torch.bfloat16,
    load_in_4bit = False, 
    )
    model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized at 0
    bias = "none", # No additional bias terms
    use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
    random_state = 3407,
    use_rslora = False, # Rank stabilized LoRA, can be enabled for stability
    )
    return model, tokenizer, max_seq_length

def getweight(model):
    netstruc =[]
    for name, param in model.named_parameters():
        # if 'layers.0' in name:
        #     print(name,param.shape)
            # netstruc.append(param.grad)
        # if param.requires_grad:
        netstruc.append(name+','+str(param.shape))
    with open('weight-tmp.txt', 'w') as f:
        for item in netstruc:
            f.write("%s\n" % item)
    return 0

alpaca_prompt = """下面是一项描述任务的说明，配有提供进一步背景信息的输入。写出一个适当完成请求的回应。
### Instruction:
{}
### Input:
{}
### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts,}

def getdataset():
    dataset = load_dataset("json", data_files=['./qadolly.json'], split="train")
    dataset = dataset.map(formatting_prompts_func,batched = True,)
    return dataset

class MyTrainer7(SFTTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model0  = getmodel0()
    #     self.step = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        # for name, param in model.named_parameters():
        #     if 'self_attn.' in name:
        #         param.requires_grad = flag
        #     elif ('mlp.gate_proj' in name) or ('mlp.up_proj' in name) or ('mlp.down_proj' in name):
        #         param.requires_grad = not flag
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        loss.backward()
        if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps      
        return loss.detach() 
        
if __name__ == '__main__':
    start_time = time.time()
    model, tokenizer, max_seq_length = getmodel()
    EOS_TOKEN = tokenizer.eos_token
    dataset = getdataset()
    trainer = MyTrainer7(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 50,
    num_train_epochs=1,
    warmup_steps = 5,
    # max_steps = 10,
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "./outputs",
    ),
    )
    # trainer_stats = trainer.train()
    model0 = getmodel0()
    model0.float()
    newmodel = merge_model(trainer.model.model.model,model0,16)
    # print(trainer.model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.shape)
    trainer.model.model.model = newmodel
    # getweight(trainer.model)
    trainer_stats = trainer.train()
    trainer.save_model()
    trainer.save_state()
    end_time = time.time()
    print("训练耗时：", end_time - start_time)