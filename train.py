import os
import time
import torch.distributed as dist
from transformer import make_model,LabelSmoothing,rate, TrainState,Batch,run_epoch,SimpleLossCompute
import torch
from torch.optim.lr_scheduler import LambdaLR
from preprocess import run_preprocessing

def train_worker():
    config={
        "batch_size":64,
        "base_lr":1.0,
        "warmup":3000,
        "epochs":40,
        "n_layers":6,
        "accum_iter":1
    }
    
    if torch.cuda.is_available():
        # 强制当前进程使用 0 号显卡 (通常默认的高性能 NVIDIA 卡就是 0)
        torch.cuda.set_device(0) 
        device = torch.device("cuda:0")
        print(f"成功锁定独显: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，正在使用 CPU 训练")

    vocab_src,vocab_tgt ,train_loader=run_preprocessing()
    torch.save((vocab_src,vocab_tgt),"vocab.pt")
    print("准备并保存词表")

    #基础参数设置，初始化，构建模型
    pad_idx=vocab_tgt.blank_idx
    d_model=512
    model=make_model(
        len(vocab_src),len(vocab_tgt),N=config["n_layers"]
        )
    model.to(device)
    
    #损失函数
    criterion=LabelSmoothing(size=len(vocab_tgt),padding_idx=pad_idx,smoothing=0.1)
    criterion.to(device)
    
    optimizer=torch.optim.Adam(
        model.parameters(),lr=config["base_lr"],betas=(0.9,0.98),eps=1e-9
        )
    lr_scheduler=LambdaLR(optimizer=optimizer,lr_lambda=lambda step:rate(step,d_model,factor=1.0,warmup=config["warmup"]))
    
    train_state=TrainState()
    loss_compute=SimpleLossCompute(model.generator,criterion)

    print("开始训练循环...")
    start_time=time.time()
    for epoch in range(config["epochs"]):
        print(f"\n================= Epoch {epoch + 1}/{config['epochs']} =================")
        model.train()
        batch_iter=(
            Batch(src.to(device),tgt.to(device),pad=pad_idx)
            for src,tgt in train_loader
        )
        epoch_loss,train_state=run_epoch(
            data_iter=batch_iter,model=model,loss_compute=loss_compute,
            optimizer=optimizer,scheduler=lr_scheduler,mode="train+log",
            accum_iter=config["accum_iter"],train_state=train_state
        )
        print(f"Epoch {epoch+1} 结束 | 平均Token损失: {epoch_loss:.4f}")

        #保存当前Epoch的权重
        os.makedirs("checkpoints",exist_ok=True)
        save_path=f"checkpoints/model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(),save_path)
        print(f"模型权重已保存至: {save_path}")
    total_time=(time.time() - start_time) / 60
    print(f"\n训练完成!总耗时: {total_time:.2f} 分钟.")

if __name__=="__main__":
    train_worker()

# print(f"Train worker process using GPU:{gpu} for training,flush=True")
    # torch.cuda.set_device(gpu)
# module=model
# is_main_process=True    
# if is_distributed:
    #     dist.init_process_qroup("nccl",init_method="env://",rank=gpu,world_size=ngpus_per_node)
    #     model=DDP(model,device_id=[gpu])
    #     module=model.module
    #     is_main_process = gpu == 0
