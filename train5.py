#有提高，三层网络，大概72轮，mmlu能提高到44.98%
import torch
import time
from unsloth import FastLanguageModel,is_bfloat16_supported
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt,train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import copy
EOS_TOKEN = ''


def merge_model(model,old_model):
    for i in range(16):
        if i<16:
            model.model.layers[i].self_attn.q_proj.lora_A.default.weight = torch.nn.Parameter((old_model.model.model.layers[i].self_attn.q_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.q_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.q_proj.lora_A.default.weight)/3)
            model.model.layers[i].self_attn.q_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.q_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.q_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.q_proj.lora_B.default.weight)/3)
            model.model.layers[i].self_attn.k_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.k_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.k_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.k_proj.lora_A.default.weight)/3)
            model.model.layers[i].self_attn.k_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.k_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.k_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.k_proj.lora_B.default.weight)/3)
            model.model.layers[i].self_attn.v_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.v_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.v_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.v_proj.lora_A.default.weight)/3)
            model.model.layers[i].self_attn.v_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.v_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.v_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.v_proj.lora_B.default.weight)/3)
            model.model.layers[i].self_attn.o_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.o_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.o_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.o_proj.lora_A.default.weight)/3)
            model.model.layers[i].self_attn.o_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.o_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.o_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.o_proj.lora_B.default.weight)/3)
            model.model.layers[i].mlp.gate_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.gate_proj.lora_A.default.weight+old_model.model.model.layers[i+16].mlp.gate_proj.lora_A.default.weight+old_model.model.model.layers[i+32].mlp.gate_proj.lora_A.default.weight)/3)
            model.model.layers[i].mlp.gate_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.gate_proj.lora_B.default.weight+old_model.model.model.layers[i+16].mlp.gate_proj.lora_B.default.weight+old_model.model.model.layers[i+32].mlp.gate_proj.lora_B.default.weight)/3)
            model.model.layers[i].mlp.up_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.up_proj.lora_A.default.weight+old_model.model.model.layers[i+16].mlp.up_proj.lora_A.default.weight+old_model.model.model.layers[i+32].mlp.up_proj.lora_A.default.weight)/3)
            model.model.layers[i].mlp.up_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.up_proj.lora_B.default.weight+old_model.model.model.layers[i+16].mlp.up_proj.lora_B.default.weight+old_model.model.model.layers[i+32].mlp.up_proj.lora_B.default.weight)/3)
            model.model.layers[i].mlp.down_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.down_proj.lora_A.default.weight+old_model.model.model.layers[i+16].mlp.down_proj.lora_A.default.weight+old_model.model.model.layers[i+32].mlp.down_proj.lora_A.default.weight)/3)
            model.model.layers[i].mlp.down_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.down_proj.lora_B.default.weight+old_model.model.model.layers[i+16].mlp.down_proj.lora_B.default.weight+old_model.model.model.layers[i+32].mlp.down_proj.lora_B.default.weight)/3)
            # model.model.layers[i].self_attn.q_proj.lora_A.default.weight = torch.nn.Parameter((old_model.model.model.layers[i].self_attn.q_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.q_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.q_proj.lora_A.default.weight+old_model.model.model.layers[i+48].self_attn.q_proj.lora_A.default.weight)/4)
            # model.model.layers[i].self_attn.q_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.q_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.q_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.q_proj.lora_B.default.weight+old_model.model.model.layers[i+48].self_attn.q_proj.lora_B.default.weight)/4)
            # model.model.layers[i].self_attn.k_proj.lora_A.default.weight = torch.nn.Parameter((old_model.model.model.layers[i].self_attn.k_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.k_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.k_proj.lora_A.default.weight+old_model.model.model.layers[i+48].self_attn.k_proj.lora_A.default.weight)/4)
            # model.model.layers[i].self_attn.k_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.k_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.k_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.k_proj.lora_B.default.weight+old_model.model.model.layers[i+48].self_attn.k_proj.lora_B.default.weight)/4)            
            # model.model.layers[i].self_attn.v_proj.lora_A.default.weight = torch.nn.Parameter((old_model.model.model.layers[i].self_attn.v_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.v_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.v_proj.lora_A.default.weight+old_model.model.model.layers[i+48].self_attn.v_proj.lora_A.default.weight)/4)
            # model.model.layers[i].self_attn.v_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.v_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.v_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.v_proj.lora_B.default.weight+old_model.model.model.layers[i+48].self_attn.v_proj.lora_B.default.weight)/4)
            # model.model.layers[i].self_attn.o_proj.lora_A.default.weight = torch.nn.Parameter((old_model.model.model.layers[i].self_attn.o_proj.lora_A.default.weight+old_model.model.model.layers[i+16].self_attn.o_proj.lora_A.default.weight+old_model.model.model.layers[i+32].self_attn.o_proj.lora_A.default.weight+old_model.model.model.layers[i+48].self_attn.o_proj.lora_A.default.weight)/4)
            # model.model.layers[i].self_attn.o_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].self_attn.o_proj.lora_B.default.weight+old_model.model.model.layers[i+16].self_attn.o_proj.lora_B.default.weight+old_model.model.model.layers[i+32].self_attn.o_proj.lora_B.default.weight+old_model.model.model.layers[i+48].self_attn.o_proj.lora_B.default.weight)/4)
            # model.model.layers[i].mlp.gate_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.gate_proj.lora_A.default.weight+old_model.model.model.layers[i+16].mlp.gate_proj.lora_A.default.weight+old_model.model.model.layers[i+32].mlp.gate_proj.lora_A.default.weight+old_model.model.model.layers[i+48].mlp.gate_proj.lora_A.default.weight)/4)
            # model.model.layers[i].mlp.gate_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.gate_proj.lora_B.default.weight+old_model.model.model.layers[i+16].mlp.gate_proj.lora_B.default.weight+old_model.model.model.layers[i+32].mlp.gate_proj.lora_B.default.weight+old_model.model.model.layers[i+48].mlp.gate_proj.lora_B.default.weight)/4)
            # model.model.layers[i].mlp.up_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.up_proj.lora_A.default.weight+old_model.model.model.layers[i+16].mlp.up_proj.lora_A.default.weight+old_model.model.model.layers[i+32].mlp.up_proj.lora_A.default.weight+old_model.model.model.layers[i+48].mlp.up_proj.lora_A.default.weight)/4)
            # model.model.layers[i].mlp.up_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.up_proj.lora_B.default.weight+old_model.model.model.layers[i+16].mlp.up_proj.lora_B.default.weight+old_model.model.model.layers[i+32].mlp.up_proj.lora_B.default.weight+old_model.model.model.layers[i+48].mlp.up_proj.lora_B.default.weight)/4)
            # model.model.layers[i].mlp.down_proj.lora_A.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.down_proj.lora_A.default.weight+old_model.model.model.layers[i+16].mlp.down_proj.lora_A.default.weight+old_model.model.model.layers[i+32].mlp.down_proj.lora_A.default.weight+old_model.model.model.layers[i+48].mlp.down_proj.lora_A.default.weight)/4)
            # model.model.layers[i].mlp.down_proj.lora_B.default.weight =  torch.nn.Parameter((old_model.model.model.layers[i].mlp.down_proj.lora_B.default.weight+old_model.model.model.layers[i+16].mlp.down_proj.lora_B.default.weight+old_model.model.model.layers[i+32].mlp.down_proj.lora_B.default.weight+old_model.model.model.layers[i+48].mlp.down_proj.lora_B.default.weight)/4)    
        else:
             model.model.layers[i] = old_model.model.model.layers[i]
    return model


def getmodel0():
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/src/models/llama32-1b",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Reduce memory usage with 4-bit quantization
    dtype =  torch.bfloat16
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
    model_name = "/src/models/qwen7b",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Reduce memory usage with 4-bit quantization
    dtype =  torch.bfloat16
    )
    print(model)
    # for j in range(1,3):
    #     for i in range(16):
    #         newlayer =copy.deepcopy(model.model.layers[i]) 
    #         model.model.layers.add_module(str(j*16+i), newlayer)

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
    # return model, tokenizer, max_seq_length

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

class MyTrainer4(SFTTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        # self.step += 1
        # flag = False
        # if self.step % 3 == 0:
        #     flag = True
        # for name, param in model.named_parameters():
        #     if 'self_attn.' in name:
        #         param.requires_grad = not flag
        #     elif ('mlp.gate_proj' in name) or ('mlp.up_proj' in name) or ('mlp.down_proj' in name):
        #         param.requires_grad =  flag
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
    getmodel()
    # start_time = time.time()
    # model, tokenizer, max_seq_length = getmodel()
    
    # EOS_TOKEN = tokenizer.eos_token
    # dataset = getdataset()
    # trainer = MyTrainer4(
    # model = model,
    # tokenizer = tokenizer,
    # train_dataset = dataset,
    # dataset_text_field = "text",
    # max_seq_length = max_seq_length,
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    # dataset_num_proc = 2,
    # packing = False,
    # args = TrainingArguments(
    # per_device_train_batch_size = 2,
    # gradient_accumulation_steps = 4,
    # warmup_steps = 5,
    # max_steps = 36,
    # learning_rate = 2e-4,
    # fp16 = not is_bfloat16_supported(),
    # bf16 = is_bfloat16_supported(),
    # logging_steps = 1,
    # optim = "adamw_8bit",
    # weight_decay = 0.01,
    # lr_scheduler_type = "linear",
    # seed = 3407,
    # output_dir = "./outputs",
    # ),
    # )
    # trainer_stats = trainer.train()
    # # print(trainer.model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight)
    # model0 = getmodel0()
    # newmodel = merge_model(model0.model, trainer.model)
    # trainer.model.model.model = newmodel.model
    # # print(trainer.model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight)
    # # getweight(trainer.model)
    # trainer.save_model()
    # trainer.save_state()
    # end_time = time.time()
    # print("训练耗时：", end_time - start_time)