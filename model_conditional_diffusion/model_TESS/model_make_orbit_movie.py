'''
This file makes an animation of predicted scattered light for a model.
'''

from utils import *

# # @title TESS Dataset
class TESSDataset(Dataset):
    def __init__(self, angle_filename, ccd_folder, image_shape, orbits_to_include, num_processes=20):
        start_time = time.time()
        
        # get data
        self.angle_folder = "/pdo/users/jlupoiii/TESS/data/angles/"
        self.ccd_folder = ccd_folder
        self.orbits_to_include = orbits_to_include
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
                if int(self.angles_dic[filename[18:18+8]]['orbit']) in self.orbits_to_include:
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


# MOVIE MAKING

# hardcoding these here - make sure these match the parameters of the model being loaded
# model parameters
n_T = 600
n_feat = 256
eval_folder = 'eval/'
predictions_folder = 'predictions/'

# dataset parameters
model_dir = 'model_TESS_O11-54_im64x64_multipleGPUs_splitOrbits_earlyStop/'
os.makedirs(os.path.join(model_dir, eval_folder), exist_ok=True) # makes eval folder
os.makedirs(os.path.join(model_dir, eval_folder, predictions_folder), exist_ok=True) # makes predictions folder
angle_filename = "angles_O11-54_data_dic.pkl"
ccd_folder = "/pdo/users/jlupoiii/TESS/data/processed_images_im64x64/"
image_shape = (64,64)
num_processes = 40
model_state_dic_filename = "model_epoch423.pth"
N = 10
orbits_to_predict = [13]
orbits_to_animate = [13]
movie_name = 'O13_movie.gif'

# loads dataset
tess_dataset = TESSDataset(angle_filename, ccd_folder, image_shape, orbits_to_predict, num_processes)

# the following 2 functions (setup, predict) serve to make and save predictions for ffi's using preprocessing
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def predict(rank, world_size):

    # setting up models for each GPU
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Creates dataloader, different for each GPU so not repeating predicitons
    sampler = DistributedSampler(tess_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(tess_dataset, batch_size=1, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)

    print(f"GPU of rank {rank} has {len(dataloader)} images to predict")

    # define model
    in_dim = next(iter(dataloader))['x'].shape[2]
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, in_dim=in_dim,n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    
    # load a model
    ddpm.load_state_dict(torch.load(os.path.join(model_dir, model_state_dic_filename)))
    ddpm.to(device)
    print('model directory:', os.path.join(model_dir, model_state_dic_filename))

    torch.cuda.empty_cache()
    ddpm.eval()
    with torch.no_grad():
        for data_dic in dataloader:
            torch.cuda.empty_cache()

            # loading a datapoint
            ffi_num = data_dic['ffi_num'][0]
            orbit = data_dic['orbit'][0]
            x = data_dic['y'].to(device) # .to(device) has to do with CPU and the GPU
            c = data_dic['x'].to(device)

            # generate samples using model, each sample is an x_i
            x_gen, x_gen_store = ddpm.sample_c(c, N, (1, image_shape[0], image_shape[1]), device)

            # taking mean and std dev of all predictions (downsampled)
            bar_x = torch.mean(x_gen.cpu(), dim=0) # Mean prediction for model's predictions

            # upscale generated samples. Using linear interpolation
            order = 1 # degree of interpolation. 1 is linear
            x_gen_upsampled = scipy.ndimage.zoom(x_gen.cpu(), (1, 1, 4096//image_shape[0], 4096//image_shape[0]), order=order)
            x_gen_upsampled = torch.tensor(x_gen_upsampled)

            # taking mean of all upsampled predictions
            bar_x_upsampled = torch.mean(x_gen_upsampled, dim=0) # Mean prediction for upsampled predictions

            # pickles and saves file to os.path.join(model_dir, eval_folder, predictions_folder)
            filepath = os.path.join(model_dir, eval_folder, predictions_folder, f'O{orbit}_{ffi_num}_N{N}_prediciton.pkl')
            with open(filepath, 'wb') as file:
                pickle.dump(np.array(bar_x_upsampled[0]), file)
                print(f'predicted and saved orbit {orbit}\'s ffi{ffi_num} to {filepath}')


def make_movie():
    # dictionary maps ffi_num to the prediction image
    ffi_to_pred_dic = {}
    for filename in os.listdir(os.path.join(model_dir, eval_folder, predictions_folder)):
        orbit = filename[1:3]
        if int(orbit) not in orbits_to_animate: continue # if not in the orbit we want to animate, skip
        ffi_num = filename[4:12]
        with open(os.path.join(model_dir, eval_folder, predictions_folder, filename), 'rb') as filename:
            # unpickle and save predictions image to the dictionary
            ffi_to_pred_dic[ffi_num] = pickle.load(filename)
            
    # sorts images with respect to ffi_num
    sorted_frames = [ffi_to_pred_dic[ffi_num] for ffi_num in sorted(ffi_to_pred_dic.keys())]

    # makes fig which represents the frames in an animation
    fig, axs = plt.subplots(nrows=1, ncols=1,sharex=True,sharey=True, figsize=(8,8))
    # animate_diff is a helper function for the FuncAnimation fuctions. Converts the image at index i to a frame in the animation
    def animate_diff(i, sorted_frames):
        print(f'gif animating frame {i} of {len(sorted_frames)}', end='\r')
        img = axs.imshow(sorted_frames[i], cmap='gray', vmin=0, vmax=1)
        axs.set_xticks([])
        axs.set_yticks([])
        return img,
    # generates the animation based on sorted_frames using the animate_diff helper function
    ani = FuncAnimation(fig, animate_diff, fargs=[sorted_frames], interval=200, blit=False, repeat=True, frames=len(sorted_frames))

    # saves gif
    save_file = os.path.join(model_dir, eval_folder, movie_name)
    ani.save(save_file, writer=ImageMagickWriter(fps=15))
    print(f'Done animating movie and saved to {os.path.join(model_dir, eval_folder, movie_name)}')



if __name__ == "__main__":
    # # run to save predictions
    # n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    # world_size = n_gpus
    # print(f'world size: {world_size}')
    # mp.spawn(predict, args=(world_size,), nprocs=world_size, join=True)

    # run to make movie out of already-saved predictions
    make_movie()










