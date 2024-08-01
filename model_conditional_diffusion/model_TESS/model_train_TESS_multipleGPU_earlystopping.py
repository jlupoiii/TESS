''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from utils import *

# # @title TESS Dataset
class TESSDataset(Dataset):
    def __init__(self, angle_filename, ccd_folder, image_shape, num_processes=20):
        start_time = time.time()
        
        # get data
        self.angle_folder = "/pdo/users/jlupoiii/TESS/data/angles/"
        self.ccd_folder = ccd_folder
        # self.ccd_folder = "/pdo/users/jlupoiii/TESS/data/processed_images_im512x512/"
        self.image_shape = image_shape
        
        # Create a pool of processes
        pool = multiprocessing.Pool(processes=num_processes)

        # data matrices
        # X = []
        # Y = []
        # ffi_nums = []
        self.data = []
        self.labels = []
        self.ffi_nums = []

        self.angles_dic = pickle.load(open(self.angle_folder+angle_filename, "rb"))

        files = []
        for filename in os.listdir(self.ccd_folder):
            if filename[18:18+8] in self.angles_dic.keys():
                files.append(filename)
    

        pbar_files = tqdm(files)
        results = []
        # print(f"About to start loading images to a list at time {(time.time() - start_time):.2f}")
        results = pool.map(self.load_images_worker, pbar_files)
        # print(f"made list of all processed results at time {(time.time() - start_time):.2f}")

        # Process the results
        pbar_results = tqdm(results)
        for x, y, ffi_num in pbar_results:
            if x is not None:
                self.data.append(x)
                self.labels.append(y)
                self.ffi_nums.append(ffi_num)
        
        pool.close()
        pool.join()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time to build dataset of {len(self.data)} points: {total_time:.2f} seconds")

    def load_images_worker(self, filename):
        if len(filename) < 40 or filename[27] != '3': 
            return None, None, None

        image_arr = pickle.load(open(self.ccd_folder + filename, "rb"))

        
        ffi_num = filename[18:18+8]
        try:
            angles = self.angles_dic[ffi_num]
        except KeyError:
            return None, None, None
            
        x = np.array([angles['1/ED'], angles['1/MD'], angles['1/ED^2'], angles['1/MD^2'], angles['Eel'], angles['Eaz'], angles['Mel'], angles['Maz'], angles['E3el'], angles['E3az'], angles['M3el'], angles['M3az']]) # for inputs of 12 values
        # x = np.array([angles['1/ED'], angles['1/MD'], angles['1/ED^2'], angles['1/MD^2'], angles['Eel'], angles['Eaz'], angles['Mel'], angles['Maz']]) # for inputs of 8 values
        x = Image.fromarray(x)
        y = image_arr.flatten()
        y = Image.fromarray(y)

        return x, y, ffi_num


    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        angles_image = self.data[idx]
        ffi_image = self.labels[idx]
        ffi_num = self.ffi_nums[idx]
        orbit = self.angles_dic[ffi_num]["orbit"]

        # print('hereeee', angles_image.size)
        # print('hereeee2', type(angles_image.size))

        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda s: s.reshape(1, (angles_image.size)[1]) # for inputs of any number of values
        ])
        target_transform = transforms.Compose([
            lambda s: np.array(s),
            # lambda s: s.reshape((1024,1024)),
            lambda s: s.reshape(self.image_shape),
            transforms.ToTensor()
        ])

        angles_image = transform(angles_image)
        ffi_image = target_transform(ffi_image)

        # X: 1x12 vector of angles and distances
        # Y: 16x16 image (or other image size)
        return {"x":angles_image, "y":ffi_image, "ffi_num": ffi_num, "orbit": orbit}



# MODEL TRAINING

# hardcoding these here
n_epoch = 1000
batch_size = 16
train_ratio = 0.8
n_T = 600 # 400
n_feat = 256 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = True
epoch_checkpoint = 100
checkpoint_gpu = 0
patience = 150 # for early stopping

# dataset parameters
save_dir = 'model_TESS_O11-54_im64x64_multipleGPUs_earlyStop_TEST/'
os.makedirs(save_dir, exist_ok=True)
angle_filename = "angles_O11-54_data_dic.pkl"
ccd_folder = "/pdo/users/jlupoiii/TESS/data/processed_images_im64x64/"
image_shape = (64,64)
num_processes = 80

# saves txt file with info about model parameters
with open(os.path.join(save_dir, 'model_info.txt'), 'w') as f:
    f.write("MODEL AND DATA INFORMATION\n")
    f.write(f"Max number of epochs\t\t\t{n_epoch}\n")
    f.write(f"Batch size\t\t\t\t\t\t{batch_size}\n")
    f.write(f"training/validation ratio\t\t{train_ratio}\n")
    f.write(f"n_T (diffusion timesteps)\t\t{n_T}\n")
    f.write(f"n_feat (CNN num of features)\t{n_feat}\n")
    f.write(f"Learning rate\t\t\t\t\t{lrate}\n")
    f.write(f"Image size\t\t\t\t\t\t{image_shape}\n")
    f.write(f"GPUs used\t\t\t\t\t\t{torch.cuda.device_count()}")
    print('saved model info')


tess_dataset = TESSDataset(angle_filename, ccd_folder, image_shape, num_processes)

# # randomly separate data into training and validation sets
# # train:20768, valid:5192 - random
# num_train_samples = int(train_ratio * len(tess_dataset))
# num_valid_samples = len(tess_dataset) - num_train_samples
# train_dataset, valid_dataset = random_split(tess_dataset, [num_train_samples, num_valid_samples])

# separate data into training and validation sets by orbit. training: 11-46, validation: 47-54
# train:21021, valid:4939 - by orbit, <= 47 and > 47
train_indices = [idx for idx, data_point in enumerate(tess_dataset) if int(data_point["orbit"]) <= 47]
valid_indices = [idx for idx, data_point in enumerate(tess_dataset) if int(data_point["orbit"]) > 47]
train_dataset = Subset(tess_dataset, train_indices)
valid_dataset = Subset(tess_dataset, valid_indices)
num_train_samples = len(train_indices)
num_valid_samples = len(valid_indices)

# saving datapoints that are in training set vs validation set.
training_dataset_ffis = [train_dataset[index]['ffi_num'] for index in range(len(train_dataset))]
validation_dataset_ffis = [valid_dataset[index]['ffi_num'] for index in range(len(valid_dataset))]
with open(os.path.join(save_dir, 'training_dataset_ffinumbers.pkl'), 'wb') as file:
    pickle.dump(training_dataset_ffis, file)
    print(f"saved training dataset to {file}")
with open(os.path.join(save_dir, 'validation_dataset_ffinumbers.pkl'), 'wb') as file:
    pickle.dump(validation_dataset_ffis, file)
    print(f"saved validation dataset to {file}")

print(f'Full dataset has {num_train_samples+num_valid_samples} datapoints')
print(f'Training dataset has {num_train_samples} datapoints')
print(f'Validation dataset has {num_valid_samples} datapoints')
print(f"x-shape:, {tess_dataset[0]['x'].shape}")
print(f"y-shape:, {tess_dataset[0]['y'].shape}")
print()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train(rank, world_size):
    
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')

    # Create training dataloader, different for each GPU
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=True, shuffle=False, sampler=train_sampler)

    # Create validation dataloader, different for each GPU
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=True, shuffle=False, sampler=valid_sampler)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=True, shuffle=False, sampler=valid_sampler)

    
    print(f"GPU of rank {rank} has {len(train_dataloader)} training batches and {len(valid_dataloader)} validation batches")

    # Create model and optimizer
    in_dim = next(iter(valid_dataloader))['x'].shape[2]
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, in_dim = in_dim, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm = nn.parallel.DistributedDataParallel(ddpm, device_ids=[rank])
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    loss_history_train = []
    loss_history_valid = []
    time_history = []
    start_training_time = time.time()
    
    # Training loop
    for ep in range(n_epoch):
        print(f'epoch {ep} training, GPU rank {rank}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar_train = tqdm(train_dataloader)
        loss_ema_train = None

        for data_dic_train in pbar_train:
            optim.zero_grad()
            x_train = data_dic_train['y'].to(device)
            c_train = data_dic_train['x'].to(device)
            ffi_nums_train = data_dic_train['ffi_num']
            orbits_train = data_dic_train['orbit']

            # print('size I have here needs to be > 10 and is', len(x_train))

            loss_train = ddpm(x_train, c_train)
            loss_train.backward()
            if loss_ema_train is None:
                loss_ema_train = loss_train.item()
            else:
                loss_ema_train = 0.95 * loss_ema_train + 0.05 * loss_train.item()
            pbar_train.set_description(f"training loss: {loss_ema_train:.4f}")
            optim.step()

        loss_history_train.append(loss_ema_train)
        
        # for eval, save an image of rows of datapoint predictions. The first column are the real
        # images and the rest are predictions
        ddpm.eval()
        with torch.no_grad():
            # calculating validation loss for the epoch
            print(f'epoch {ep} validation, GPU rank {rank}')
            pbar_valid = tqdm(valid_dataloader)
            loss_ema_valid = None
            for data_dic_valid in pbar_valid:
                x_valid = data_dic_valid['y'].to(device)
                c_valid = data_dic_valid['x'].to(device)
                ffi_nums_valid = data_dic_valid['ffi_num']
                orbits_valid = data_dic_valid['orbit']
    
                loss_valid = ddpm(x_valid, c_valid)
                if loss_ema_valid is None:
                    loss_ema_valid = loss_valid.item()
                else:
                    loss_ema_valid = 0.95 * loss_ema_valid + 0.05 * loss_valid.item()
                pbar_valid.set_description(f"validation loss: {loss_ema_valid:.4f}")
            loss_history_valid.append(loss_ema_valid)

            # keeps track of time
            time_history.append(round((time.time() - start_training_time)/3600, 3))

            # condidions for early stopping. If no improvement has been seen in validation loss in 'patience' num epochs
            not_improving = (loss_history_valid[-1*patience:][0] == min(loss_history_valid[-1*patience:]))
            all_training_below_valid = all(t <= v for t, v in zip(loss_history_train[-1*patience:], loss_history_valid[-1*patience:]))
            stop_early = not_improving and all_training_below_valid

            
            # print(f'Will we run the checkpoint? This is epoch {ep}. Rank is {rank}, {type(rank)}. GPU is {checkpoint_gpu}, {type(checkpoint_gpu)}, where our condition is {rank==checkpoint_gpu}')
            if (ep%epoch_checkpoint==0 or ep == int(n_epoch-1) or stop_early) and rank==checkpoint_gpu: # for multiple GPUs, only do predictions on GPU with rank <checkpoint_gpu> to only do predictions once.

                print(f'Running checkpoint at epoch {ep} on GPU {checkpoint_gpu}')
                
                n_datapoint = min(10, batch_size) # want at most 10 and at least batch_size datapoints
                n_sample = 5
                
                def sample_save_plots(x_real, c_real, ffi_nums_real, orbits_real, train_or_valid_string):

                    print(f'running checkpoint with GPU rank {rank}')
                    
                    # want each row to be for one datapoint, and for there to be n_sample columns
                    # want the first column to be the real image

                    # choose the first n_datapoint datapoints to do predictions on
                    # The dataloader has shuffle=True so these datapoints are always random

                    x_gen, x_gen_store = ddpm.module.sample_c(c_real, n_sample, (1, image_shape[0], image_shape[1]), device)
                
                    x_all = torch.Tensor().to(device)
                    for i in range(n_datapoint):
                        x_all = torch.cat([x_all, x_real[i:i+1], x_gen[i*n_sample:(i+1)*n_sample]])

                    fig, axes = plt.subplots(n_datapoint, n_sample+1, figsize=(15, 30))
                    plt.subplots_adjust(top=1.7)
                    for idx in range(x_all.shape[0]):
                        image = x_all[idx, 0, :, :].cpu().detach().numpy()
                        axes[idx//(n_sample+1), idx%(n_sample+1)].imshow(image, cmap='gray', vmin=0, vmax=1)
                        axes[idx//(n_sample+1), idx%(n_sample+1)].axis('off')

                    # set labels for sampled columns
                    for i in range(n_sample):
                        axes[0, i+1].set_title(f"Sample {i+1} \n ", fontsize=12)\

                    # set labels for each datapoint
                    for j in range(n_datapoint):
                        # print(f'here we should have orbits_real:{len(orbits_real)} n_datapoint:{n_datapoint}')
                        data_title = f"O{orbits_real[j]} , ffi {ffi_nums_real[j]}"
                        if j==0: data_title = f"Original\n{data_title}"
                        axes[j, 0].set_title(data_title, fontsize=12)

                    # Sets title for whole figure
                    fig.suptitle(f"{train_or_valid_string} predictions for epoch {ep}", fontsize = 25)

                    # save images
                    plt.tight_layout()
                    fig.savefig(save_dir + f"image_ep{ep}_{train_or_valid_string.lower()}.pdf")
                    print('saved image at ' + save_dir + f"image_ep{ep}_{train_or_valid_string.lower()}.pdf")
                    plt.close()

                # training set
                x_train_real = x_train[:n_datapoint]
                c_train_real = c_train[:n_datapoint]
                ffi_nums_train_real = ffi_nums_train[:n_datapoint]
                orbits_train_real = orbits_train[:n_datapoint]
                sample_save_plots(x_train_real, c_train_real, ffi_nums_train_real, orbits_train_real, "Training")

                # validation set
                x_valid_real = x_valid[:n_datapoint]
                c_valid_real = c_valid[:n_datapoint]
                ffi_nums_valid_real = ffi_nums_valid[:n_datapoint]
                orbits_valid_real = orbits_valid[:n_datapoint]
                sample_save_plots(x_valid_real, c_valid_real, ffi_nums_valid_real, orbits_valid_real, "Validation")

                # save loss graph
                plt.plot(loss_history_valid, label="Validation Loss")
                plt.plot(loss_history_train, label="Training Loss")
                plt.xlabel('Epoch')
                plt.ylabel('MSE Loss')
                plt.title('Training and Validation MSE Loss Over Epochs')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
                plt.close()

                # save only last 50 epochs of loss graph
                if len(loss_history_valid) >= 50:
                    plt.plot(range(len(loss_history_valid)-50, len(loss_history_valid)), loss_history_valid[-50:], label="Validation Loss")
                    plt.plot(range(len(loss_history_valid)-50, len(loss_history_valid)), loss_history_train[-50:], label="Training Loss")
                    plt.xlabel('Epoch')
                    plt.ylabel('MSE Loss')
                    plt.title('Training and Validation MSE Loss Over Last 50 Epochs')
                    plt.legend()
                    plt.savefig(os.path.join(save_dir, 'loss_graph_last50.png'))
                    plt.close()

                
                # save loss txt file
                with open(os.path.join(save_dir, 'loss_history.txt'), 'w') as f:
                    f.write("Epoch\tTraining MSE Loss\tValidation MSE Loss\t\tTime(Hrs)\n")
                    # Iterate over the indices of the lists
                    for i in range(len(loss_history_valid)):
                        # Write the index and corresponding elements from the lists
                        f.write(f"{i}\t\t{'{:.4e}'.format(loss_history_train[i])}\t\t{'{:.4e}'.format(loss_history_valid[i])}\t\t{time_history[i]}\n")
                        
                # optionally save model
                if save_model:
                    torch.save(ddpm.module.state_dict(), save_dir + f"model_epoch{ep}.pth")
                    print('saved model at ' + save_dir + f"model_epoch{ep}.pth")

                # handles stopping early
                if stop_early:
                    break
                    

        
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    print(f'world size: {world_size}')
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)





