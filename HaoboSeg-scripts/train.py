from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    torch.cuda.set_device(gpu)

    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    writer = SummaryWriter(comment=f"noAtt_seed{seed}_determ_nearest_n{BATCH_SIZE}_l{weight_decay}", flush_secs=1)
    optim.param_groups[0]['lr']=1e-4
    min_loss = 0
    no_improve_step = 0

    idx = 0
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # writer = None
    lambda1 = lambda epoch : (1 - epoch/EPOCHS) **0.9 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        model.train()
        coarse_model.train()
        train_losses, train_dice_0s, train_dice_1s = [], [], []
        val_losses, val_dice_2s = [], []
        train_start = time.time()

        #training begins
        for idx, (x_batch_train, y_batch_train, _) in enumerate(train_dataloader):
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
            train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.to(device, non_blocking=True), y_batch_train.to(device, non_blocking=True))
            train_losses.append(train_loss)
            train_dice_0s.append(train_dice_0)
            train_dice_1s.append(train_dice_1)
        
        # gather mean criteria
        train_loss = np.mean(train_losses)
        train_dice_0 = np.mean(train_dice_0s)
        train_dice_1 = np.mean(train_dice_1s)

        train_end = time.time()
        train_step_t = (train_end - train_start) / (idx + 1)

        # track epochs with no improvements in training loss

        

        print(f"EPOCH: {epoch} Train_loss:  {train_loss:.4f}, Train_dice_myo: {train_dice_0:.4f}, Train_dice_chamber: {train_dice_1:.4f}, lr: {optim.param_groups[0]['lr']}")
    
        model.eval()
        coarse_model.eval()
        eval_start = time.time()
        for idx, (x_batch_val, y_batch_val, _) in enumerate(val_dataloader):
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
            val_loss, val_dice2 = val_step(x_batch_val.to(device, non_blocking=True), y_batch_val.to(device, non_blocking=True))
            val_losses.append(val_loss)
            # val_dice_0s.append(val_dice0)
            # val_dice_1s.append(val_dice1)
            val_dice_2s.append(val_dice2)
        eval_end = time.time()
        eval_step_t = (eval_end - eval_start) / (idx + 1)
        val_loss = np.mean(val_losses)
        # val_dice_0 = np.mean(val_dice_0s)
        # val_dice_1 = np.mean(val_dice_1s)
        val_dice_2 = np.mean(val_dice_2s)
        scheduler.step()
        print(f"EPOCH: {epoch} Val_loss:  {val_loss:.4f},Val_dice_whole: {val_dice_2:.4f}")
        # writer.add_scalar('Loss/train', train_loss, epoch)

        # Writes scores in this epoch to tensorboard
        if writer is not None:
            writer.add_scalars('Loss', {'train':train_loss,
                                        'validation': val_loss}, epoch)
            # writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalars('Whole Dice', {'train': train_dice_0,
                                                'validation': val_dice_2}, epoch)
            # writer.add_scalars('Chamber Dice', {'train': train_dice_1,
            #                                     'validation': val_dice_1}, epoch)
        
        if not (epoch + 1) % 20:
            torch.save(coarse_model.state_dict(), f"coarse_model1_iter{epoch}_seed{seed}_nearest_n{BATCH_SIZE}_final_drop.pth")
            torch.save(model.state_dict(), f"fine_model1_iter{epoch}_seed{seed}_nearest_n{BATCH_SIZE}_final_drop.pth")

        end_time = time.time()
        print(f"Total_time for epoch {epoch:d}: {end_time - start_time:.3f}s, Train speed: {train_step_t * 1000:.3f}ms/step, Val_speed: {eval_step_t * 1000:.3f}ms/step")
